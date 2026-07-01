"""SNI-aware egress proxy for domain-level network allow-listing.

Solves the CDN IP rotation problem: iptables rules use IP addresses
resolved at rule-generation time, but CDN IPs (Fastly, Cloudflare)
rotate constantly. An SNI-aware proxy inspects the domain in the TLS
ClientHello (SNI) or HTTP Host header and allows/denies based on the
domain — not the IP.

Architecture:
  1. The proxy runs on the host (or as a sidecar container).
  2. The sandbox container routes all traffic through the proxy via
     HTTP_PROXY / HTTPS_PROXY env vars or iptables REDIRECT.
  3. For HTTPS (CONNECT): the proxy checks the CONNECT target host
     against the allow-list (the authoritative destination). If
     allowed, it connects to the resolved IP, sends '200 Connection
     Established', and tunnels bidirectionally. After tunneling begins,
     it peeks at the TLS ClientHello to extract the SNI and verifies it
     matches the CONNECT target (defense-in-depth against SNI spoofing).
     If denied, it returns 403.
  4. For HTTP: the proxy reads the Host header and checks it against
     the allow-list.

This is a transparent CONNECT proxy — it does NOT perform TLS
interception (no MITM). It only inspects the SNI (which is in
cleartext in the TLS ClientHello) and the Host header (which is in
cleartext for HTTP). The actual TLS traffic passes through untouched.

SECURITY WARNING — SNI spoofing (known limitation, no MITM):
  Because there is no TLS interception, the proxy trusts the SNI value
  in the ClientHello to decide whether a connection is allowed, then
  tunnels the raw TCP stream to whatever destination the *client*
  requests (via the CONNECT target / resolved IP). A malicious
  in-sandbox process can present an allow-listed SNI (e.g. "pypi.org")
  while issuing a CONNECT to an attacker-controlled IP, or lying about
  the SNI while connecting to a disallowed host's IP. The SNI check is
  therefore an *intent* signal, not a *destination* guarantee. To close
  this gap you would need either:
    (a) TLS MITM with a CA trusted by the sandbox (breaks end-to-end
        TLS, requires cert management), OR
    (b) DNS pinning + IP allow-listing so the proxy resolves the
        allow-listed hostname itself and connects only to the resolved
        IPs (the --dns-resolver / network allow-list path is the
        primary defense; this SNI proxy is a complementary layer).
  Treat the SNI proxy as defense-in-depth, NOT a complete egress
  boundary on its own.

Usage:
    python3 -m sandbox.sni_proxy --port 8888 \\
        --allowlist "pypi.org:443,*.pypi.org:443,files.pythonhosted.org:443"

Then configure the sandbox to use it:
    --proxy-egress 127.0.0.1:8888
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import sys
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# SNI extraction
# ---------------------------------------------------------------------------

def _is_ip_literal(host: str) -> bool:
    """Return True if `host` is an IP literal (v4 or v6), not a hostname."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def extract_sni(client_hello: bytes) -> Optional[str]:
    """Extract the Server Name Indication (SNI) from a TLS ClientHello.

    The SNI is in the cleartext part of the TLS handshake (the
    ClientHello record). We parse the TLS record header and the SNI
    extension to extract the hostname — no TLS interception needed.

    Returns the hostname string, or None if not found / not a valid
    ClientHello.
    """
    # TLS record: type(1) + version(2) + length(2) + handshake
    if len(client_hello) < 5 or client_hello[0] != 0x16:
        return None  # not a TLS handshake
    record_len = int.from_bytes(client_hello[3:5], "big")
    if len(client_hello) < 5 + record_len:
        return None  # incomplete record
    # Handshake: type(1) + length(3) + version(2) + random(32) + session_id
    handshake = client_hello[5:]
    if len(handshake) < 4 or handshake[0] != 0x01:
        return None  # not a ClientHello
    pos = 4 + 2 + 32  # skip type, length, version, random
    if pos >= len(handshake):
        return None
    # Session ID
    session_id_len = handshake[pos]
    pos += 1 + session_id_len
    # Cipher suites
    if pos + 2 > len(handshake):
        return None
    cipher_len = int.from_bytes(handshake[pos:pos+2], "big")
    pos += 2 + cipher_len
    # Compression methods
    if pos + 1 > len(handshake):
        return None
    comp_len = handshake[pos]
    pos += 1 + comp_len
    # Extensions
    if pos + 2 > len(handshake):
        return None
    ext_len = int.from_bytes(handshake[pos:pos+2], "big")
    pos += 2
    ext_end = pos + ext_len
    while pos + 4 <= ext_end and pos + 4 <= len(handshake):
        ext_type = int.from_bytes(handshake[pos:pos+2], "big")
        ext_data_len = int.from_bytes(handshake[pos+2:pos+4], "big")
        pos += 4
        if ext_type == 0x0000:  # SNI extension
            # SNI list: length(2) + entries
            if pos + 2 > len(handshake):
                return None
            sni_list_len = int.from_bytes(handshake[pos:pos+2], "big")
            sni_pos = pos + 2
            sni_end = sni_pos + sni_list_len
            while sni_pos + 3 <= sni_end and sni_pos + 3 <= len(handshake):
                name_type = handshake[sni_pos]
                name_len = int.from_bytes(handshake[sni_pos+1:sni_pos+3], "big")
                sni_pos += 3
                if name_type == 0:  # host_name
                    if sni_pos + name_len > len(handshake):
                        return None
                    return handshake[sni_pos:sni_pos+name_len].decode("ascii", errors="replace")
                sni_pos += name_len
            return None
        pos += ext_data_len
    return None


# ---------------------------------------------------------------------------
# Domain matching
# ---------------------------------------------------------------------------

def domain_matches(hostname: str, pattern: str) -> bool:
    """Check if a hostname matches an allow-list pattern.

    Supports wildcard patterns: *.example.com matches foo.example.com
    but NOT example.com or foo.bar.example.com.
    """
    hostname = hostname.lower().strip()
    pattern = pattern.lower().strip()
    if pattern.startswith("*."):
        suffix = pattern[2:]
        # *.example.com matches foo.example.com but not example.com
        # and not foo.bar.example.com (single-level wildcard)
        parts = hostname.split(".", 1)
        if len(parts) == 2 and parts[1] == suffix:
            return True
        return False
    return hostname == pattern


def _is_ip_allowed(host: str, allowed_ips: Set[str]) -> bool:
    """Check if ``host`` is an IP literal that matches an allowed IP/CIDR.

    ``allowed_ips`` may contain plain IPv4/IPv6 addresses (e.g. ``10.0.0.1``)
    or CIDR networks (e.g. ``10.0.0.0/24``). Plain hostnames are ignored; use
    :func:`is_domain_allowed` for those.

    Returns False for unparseable hosts so the caller can fall back to domain
    matching (or deny, depending on context).
    """
    if not allowed_ips:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False

    for spec in allowed_ips:
        spec = spec.strip()
        if not spec:
            continue
        try:
            network = ipaddress.ip_network(spec, strict=False)
            if addr in network:
                return True
        except ValueError:
            # Not a CIDR; try a single address.
            try:
                if addr == ipaddress.ip_address(spec):
                    return True
            except ValueError:
                continue
    return False


def is_domain_allowed(
    hostname: str,
    allowed_domains: Set[str],
    allowed_wildcards: Set[str],
    allowed_ips: Optional[Set[str]] = None,
) -> bool:
    """Check if a hostname or IP literal is allowed by the allow-list."""
    hostname = hostname.lower().strip()
    if hostname in allowed_domains:
        return True
    for pattern in allowed_wildcards:
        if domain_matches(hostname, pattern):
            return True
    if allowed_ips and _is_ip_allowed(hostname, allowed_ips):
        return True
    return False


# ---------------------------------------------------------------------------
# DNS audit logging + rate limiting (Phase 1.2)
# ---------------------------------------------------------------------------
#
# The v3.1 `--add-host` injection eliminates DNS access *inside* the
# sandbox container (no resolver in /etc/resolv.conf, only injected
# /etc/hosts entries). This closes in-container DNS-tunnel exfiltration.
#
# This class adds the second layer the egress plan calls for: a host-
# side DNS audit log + per-domain rate limiter on the proxy's OWN
# resolution path. The SNI proxy resolves allow-listed hostnames on the
# host before tunneling; logging and rate-limiting those resolutions gives
# auditability and bounds the blast radius of a compromised allow-list
# entry (e.g., a wildcard that an attacker floods with subdomain lookups).
#
# Fail-closed: when over the rate limit, the resolution is denied and the
# connection is rejected (the caller returns 403 / closes).

class DnsAuditor:
    """Per-domain DNS query audit log + sliding-window rate limiter.

    Args:
        rate_limit: max queries per domain per window. None disables
            rate limiting (logging still occurs if `log` is True).
        window: sliding window length in seconds.
        log: if True, emit one JSON line per resolution attempt to stderr
            (structured audit trail: ts, domain, ips, verdict, reason).
    """

    def __init__(
        self,
        rate_limit: Optional[int] = None,
        window: float = 60.0,
        log: bool = False,
    ):
        self.rate_limit = rate_limit
        self.window = window
        self.log = log
        self._queries: Dict[str, Deque[float]] = defaultdict(deque)
        self._denied_count = 0
        self._allowed_count = 0

    def allow_query(self, domain: str) -> bool:
        """Check the per-domain rate limit. Does NOT record a query.

        Returns True if a query for `domain` is within the rate limit
        (or if rate limiting is disabled). Returns False if the domain
        is over the limit — the caller must deny the connection.
        """
        if self.rate_limit is None:
            return True
        domain = domain.lower().strip()
        now = time.monotonic()
        bucket = self._queries[domain]
        # Evict entries outside the sliding window.
        cutoff = now - self.window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        return len(bucket) < self.rate_limit

    def record(self, domain: str, ips: List[str], verdict: str, reason: str = "") -> None:
        """Record a resolution attempt.

        Args:
            domain: the hostname being resolved.
            ips: resolved IP addresses (empty on failure/denial).
            verdict: "allowed" | "denied" | "error".
            reason: human-readable detail (e.g. "rate_limited",
                "getaddrinfo failed: ...").
        """
        domain = domain.lower().strip()
        if verdict == "allowed":
            self._allowed_count += 1
            if self.rate_limit is not None:
                self._queries[domain].append(time.monotonic())
        else:
            self._denied_count += 1
        if self.log:
            entry = {
                "ts": time.time(),
                "domain": domain,
                "ips": ips,
                "verdict": verdict,
                "reason": reason,
            }
            print(json.dumps(entry), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# SNI proxy server
# ---------------------------------------------------------------------------

class SNIEgressProxy:
    """Async SNI-aware egress proxy.

    Listens for CONNECT requests (HTTPS) and plain HTTP requests.
    For CONNECT: checks the CONNECT target host AND port against the
    allow-list, connects upstream, sends 200, then peeks at the TLS
    ClientHello SNI for defense-in-depth spoofing detection. For HTTP:
    reads the Host header and checks host AND port against the allow-list.

    Port enforcement: when an allow-list entry specifies a port (e.g.
    ``pypi.org:443``), only that port is allowed for that host. When
    no port is specified (e.g. ``pypi.org``), all ports are allowed
    (backward compat). This is tracked via ``allowed_ports``: a dict
    mapping hostname → set of allowed ports. An empty set or absent
    key means "all ports allowed".

    Allowed connections are tunneled to the destination. Denied
    connections are closed immediately.
    """

    def __init__(
        self,
        allowed_domains: Set[str],
        allowed_wildcards: Set[str],
        allowed_ips: Set[str] = None,
        allowed_ports: Optional[Dict[str, Set[int]]] = None,
        port: int = 8888,
        host: str = "127.0.0.1",
        dns_resolver: Optional[str] = None,
        dns_audit: bool = False,
        dns_rate_limit: Optional[int] = None,
        dns_rate_window: float = 60.0,
    ):
        self.allowed_domains = allowed_domains
        self.allowed_wildcards = allowed_wildcards
        self.allowed_ips = allowed_ips or set()
        # Port restrictions: hostname → set of allowed ports.
        # Empty set or absent key = all ports allowed (no restriction).
        self.allowed_ports = allowed_ports or {}
        self.port = port
        self.host = host
        self.dns_resolver = dns_resolver
        self._server: Optional[asyncio.AbstractServer] = None
        self._denied_count = 0
        self._allowed_count = 0
        # Phase 1.2: host-side DNS audit log + per-domain rate limiter.
        self._dns_auditor = DnsAuditor(
            rate_limit=dns_rate_limit,
            window=dns_rate_window,
            log=dns_audit,
        )

    def _is_port_allowed(self, host: str, port: int) -> bool:
        """Check if a port is allowed for a host.

        If the host has no port restrictions in allowed_ports (or the
        key is absent), all ports are allowed (backward compat with
        allow-list entries that don't specify a port). If the host has
        a port restriction set, the port must be in that set.

        Wildcard-aware: when there is no exact-host port restriction,
        we also check any wildcard pattern (e.g. ``*.example.com``)
        that matches the host. A wildcard entry like
        ``*.example.com:443`` stores its port restriction under the
        pattern key ``*.example.com``; without this lookup a request to
        ``foo.example.com:80`` would find no exact key and be allowed,
        defeating the port restriction.

        CIDR-aware: when the host is a concrete IP and there is no exact
        IP port restriction, we also check any CIDR prefix in allowed_ports
        that contains the IP. A CIDR entry like ``10.0.0.0/24:443`` stores
        its restriction under the key ``10.0.0.0/24``; without this lookup a
        request to ``10.0.0.5:80`` would find no exact key and be allowed.
        """
        host = host.lower().strip()
        ports = self.allowed_ports.get(host)
        if ports:
            # Exact-host port restriction present — must match.
            return port in ports
        # No exact-host restriction — check matching wildcard patterns.
        # A wildcard pattern may carry a port restriction that applies
        # to all of its matching subdomains.
        for pattern in self.allowed_wildcards:
            if domain_matches(host, pattern):
                ports = self.allowed_ports.get(pattern.lower())
                if ports:
                    return port in ports
        # CIDR-aware: if the host is an IP, check any containing CIDR.
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            addr = None
        if addr is not None:
            for key, ports in self.allowed_ports.items():
                try:
                    network = ipaddress.ip_network(key, strict=False)
                    if addr in network:
                        return port in ports
                except ValueError:
                    continue
        # No exact key and no matching wildcard/CIDR key with a restriction —
        # all ports allowed (backward compat).
        return True

    async def _resolve_host(self, host: str) -> Optional[str]:
        """Resolve a hostname with audit logging + rate limiting.

        Returns the first resolved IP string, or None if the resolution
        was denied (rate limited) or failed. The caller must deny the
        connection when None is returned (fail-closed).
        """
        # Rate-limit check BEFORE resolution (don't burn a lookup on a
        # domain that's already over the limit).
        if not self._dns_auditor.allow_query(host):
            self._dns_auditor.record(host, [], "denied", "rate_limited")
            return None
        try:
            loop = asyncio.get_event_loop()
            infos = await loop.getaddrinfo(host, None, family=socket.AF_INET)
            ips = list({info[4][0] for info in infos})
            if not ips:
                self._dns_auditor.record(host, [], "error", "no AF_INET results")
                return None
            self._dns_auditor.record(host, ips, "allowed")
            return ips[0]
        except socket.gaierror as e:
            self._dns_auditor.record(host, [], "error", f"getaddrinfo failed: {e}")
            return None

    async def start(self):
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        print(f"[SNIProxy] Listening on {self.host}:{self.port}")
        print(f"[SNIProxy] Allowed domains: {self.allowed_domains}")
        if self.allowed_wildcards:
            print(f"[SNIProxy] Allowed wildcards: {self.allowed_wildcards}")

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        print(f"[SNIProxy] Stopped "
              f"(allowed={self._allowed_count}, denied={self._denied_count})")

    async def _handle_connection(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ):
        """Handle an incoming connection from the sandbox."""
        try:
            # Read the first line to determine if it's CONNECT or HTTP.
            first_line = await asyncio.wait_for(
                client_reader.readline(), timeout=5.0
            )
            if not first_line:
                return
            line = first_line.decode("ascii", errors="replace").strip()

            if line.startswith("CONNECT "):
                await self._handle_connect(line, client_reader, client_writer)
            elif line.startswith(("GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "PATCH ")):
                await self._handle_http(line, client_reader, client_writer)
            else:
                # Unknown protocol — deny.
                client_writer.close()
                self._denied_count += 1
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass
        finally:
            try:
                client_writer.close()
            except Exception:
                pass

    async def _handle_connect(
        self,
        first_line: str,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ):
        """Handle a CONNECT request (HTTPS tunneling).

        CONNECT host:port HTTP/1.1
        <headers>

        The allow decision is based on the CONNECT target host (the
        authoritative destination the client requests). This is standard
        CONNECT proxy behavior: the client tells the proxy where it wants
        to go, the proxy checks the allow-list, connects to the remote,
        sends '200 Connection Established', and tunnels bidirectionally.

        Standard HTTPS clients wait for the 200 response BEFORE sending
        the TLS ClientHello — so we must NOT try to read the ClientHello
        before sending 200 (the previous implementation did this, which
        broke standard clients and incorrectly wrote the ClientHello
        back to the client). After sending 200, we peek at the first
        client bytes (the ClientHello) to extract the SNI and verify it
        matches the CONNECT target host (defense-in-depth against SNI
        spoofing). If the SNI is present and does NOT match, the
        connection is closed.
        """
        # Parse the target from the CONNECT line.
        # CONNECT pypi.org:443 HTTP/1.1
        parts = first_line.split()
        if len(parts) < 2:
            client_writer.close()
            self._denied_count += 1
            return
        target = parts[1]
        host, _, port = target.rpartition(":")
        if not host:
            host = target
            port = "443"

        # Check the CONNECT target host against the allow-list. This is
        # the primary allow decision — the CONNECT target is the
        # authoritative destination, available before the ClientHello.
        if not is_domain_allowed(
            host,
            self.allowed_domains,
            self.allowed_wildcards,
            self.allowed_ips,
        ):
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        # Check the CONNECT target port against the allow-list. If the
        # allow-list entry for this host specifies ports (e.g.
        # pypi.org:443), only those ports are allowed. An allow-list
        # entry without a port (e.g. pypi.org) allows all ports.
        try:
            port_num = int(port)
        except ValueError:
            port_num = 443
        if not self._is_port_allowed(host, port_num):
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        # Read and discard headers until empty line.
        while True:
            header = await asyncio.wait_for(
                client_reader.readline(), timeout=5.0
            )
            if header in (b"\r\n", b"\n", b""):
                break

        # Resolve with audit + rate limiting before connecting (Phase 1.2).
        resolved_ip = await self._resolve_host(host)
        if resolved_ip is None:
            # Resolution denied/failed — fail-closed.
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        self._allowed_count += 1
        try:
            # Connect to the real destination (resolved IP) FIRST, before
            # telling the client the tunnel is established. This way, if
            # the upstream is unreachable, we return 502 instead of 200.
            remote_reader, remote_writer = await asyncio.open_connection(
                resolved_ip, int(port)
            )
        except (ConnectionError, OSError, asyncio.TimeoutError):
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await client_writer.drain()
            return

        # Send 200 Connection Established to the client. Standard HTTPS
        # clients wait for this BEFORE sending the TLS ClientHello.
        client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await client_writer.drain()

        # Defense-in-depth: peek at the first bytes from the client (the
        # TLS ClientHello) to extract the SNI and verify it matches the
        # CONNECT target host. If the SNI is present and does NOT match,
        # close the connection (SNI spoofing attempt). If SNI is absent
        # or extraction fails, the tunnel proceeds — the CONNECT target
        # check above is the primary decision. Skip the SNI check when
        # the CONNECT target is an IP literal (no hostname to compare).
        skip_sni_check = _is_ip_literal(host)
        if not skip_sni_check:
            try:
                first_bytes = await asyncio.wait_for(
                    client_reader.read(4096), timeout=5.0
                )
            except (asyncio.TimeoutError, ConnectionError, OSError):
                remote_writer.close()
                client_writer.close()
                return
            if not first_bytes:
                # Client closed the connection immediately.
                remote_writer.close()
                return
            sni = extract_sni(first_bytes)
            if sni and sni.lower() != host.lower():
                # SNI mismatch — likely SNI spoofing. Close both sides.
                remote_writer.close()
                client_writer.close()
                return
            # Forward the peeked bytes to the remote server.
            remote_writer.write(first_bytes)
            await remote_writer.drain()

        # Tunnel bidirectionally.
        await self._tunnel(client_reader, client_writer,
                           remote_reader, remote_writer)

    async def _handle_http(
        self,
        first_line: str,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ):
        """Handle a plain HTTP request.

        The allow decision is based on the request target host:
          - An absolute URL (``GET http://host/path``) is authoritative.
          - A relative URL uses the ``Host`` header.
        If both are present, they must agree (the Host header must name
        the same host as the absolute URL). The allow-list is then
        checked against the target host and port.
        """
        # Read headers to find Host.
        headers = ""
        host_header = None
        host_header_port = None  # port from Host header (if specified)
        while True:
            header = await asyncio.wait_for(
                client_reader.readline(), timeout=5.0
            )
            if header in (b"\r\n", b"\n", b""):
                break
            header_str = header.decode("ascii", errors="replace")
            headers += header_str
            if header_str.lower().startswith("host:"):
                host_value = header_str.split(":", 1)[1].strip()
                if ":" in host_value:
                    host_header, _, hport = host_value.rpartition(":")
                    host_header_port = hport
                else:
                    host_header = host_value
                    host_header_port = None

        # Parse the URL from the request line.
        parts = first_line.split()
        url = parts[1] if len(parts) > 1 else "/"
        if url.startswith("http://"):
            # Absolute URL — extract host and path.
            url_parts = url[7:].split("/", 1)
            target_authority = url_parts[0]
            path = "/" + url_parts[1] if len(url_parts) > 1 else "/"
        else:
            target_authority = host_header
            path = url

        if target_authority is None:
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        target_host, _, target_port_str = target_authority.rpartition(":")
        if not target_host:
            target_host = target_authority
            target_port_str = ""

        # If both absolute URL and Host header are present, they must
        # agree on the host (defense against Host-header spoofing).
        if host_header is not None and url.startswith("http://"):
            if host_header.lower() != target_host.lower():
                self._denied_count += 1
                client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                await client_writer.drain()
                return

        # Effective port: URL port > Host header port > default 80.
        try:
            effective_port = (
                int(target_port_str) if target_port_str
                else int(host_header_port) if host_header_port
                else 80
            )
        except ValueError:
            effective_port = 80

        if not is_domain_allowed(
            target_host,
            self.allowed_domains,
            self.allowed_wildcards,
            self.allowed_ips,
        ):
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        if not self._is_port_allowed(target_host, effective_port):
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        # Resolve with audit + rate limiting before connecting (Phase 1.2).
        resolved_ip = await self._resolve_host(target_host)
        if resolved_ip is None:
            # Resolution denied/failed — fail-closed.
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        self._allowed_count += 1
        try:
            remote_reader, remote_writer = await asyncio.open_connection(
                resolved_ip, effective_port
            )
            # Forward the request with rewritten request line.
            # headers contains the header lines (each ending \r\n)
            # but NOT the terminating empty line — add it so the
            # remote server sees a complete HTTP request.
            remote_writer.write(f"{parts[0]} {path} HTTP/1.1\r\n".encode())
            remote_writer.write(headers.encode())
            remote_writer.write(b"\r\n")
            await remote_writer.drain()
            # Tunnel bidirectionally.
            await self._tunnel(client_reader, client_writer,
                               remote_reader, remote_writer)
        except (ConnectionError, OSError, asyncio.TimeoutError):
            pass

    async def _tunnel(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        remote_reader: asyncio.StreamReader,
        remote_writer: asyncio.StreamWriter,
    ):
        """Bidirectionally tunnel data between client and remote."""
        async def forward(src_reader, dst_writer):
            try:
                while True:
                    data = await asyncio.wait_for(src_reader.read(8192), timeout=30.0)
                    if not data:
                        break
                    dst_writer.write(data)
                    await dst_writer.drain()
            except (asyncio.TimeoutError, ConnectionError, OSError):
                pass
            finally:
                try:
                    dst_writer.close()
                except Exception:
                    pass

        await asyncio.gather(
            forward(client_reader, remote_writer),
            forward(remote_reader, client_writer),
        )


# ---------------------------------------------------------------------------
# Allow-list -> proxy set extraction
# ---------------------------------------------------------------------------

def extract_allowlist_sets(al) -> Tuple[Set[str], Set[str], Set[str], Dict[str, Set[int]]]:
    """Extract SNIEgressProxy sets from a NetworkAllowList.

    Returns ``(allowed_domains, allowed_wildcards, allowed_ips,
    allowed_ports)``.

    Wildcard handling: ``NetworkAllowList._parse_entry`` strips the
    ``*.`` prefix and stores ``entry.host="example.com"`` with
    ``entry.wildcard=True``. We reconstruct the ``*.example.com``
    pattern for ``allowed_wildcards`` and key any port restriction
    under that same pattern so ``_is_port_allowed`` can resolve it via
    ``domain_matches()``. Checking ``entry.host.startswith("*.")``
    (the old code) would NEVER be true for wildcard entries — they'd
    silently fall into the exact-domain branch, inverting the
    semantics (a wildcard-subdomain rule becomes a root-domain exact
    rule). Use ``entry.wildcard`` instead.
    """
    allowed_domains: Set[str] = set()
    allowed_wildcards: Set[str] = set()
    allowed_ips: Set[str] = set()
    allowed_ports: Dict[str, Set[int]] = {}
    for entry in al.entries:
        if entry.kind == "domain":
            if entry.wildcard:
                pattern = f"*.{entry.host}"
                allowed_wildcards.add(pattern)
                if entry.port is not None:
                    pattern_key = pattern.lower()
                    if pattern_key not in allowed_ports:
                        allowed_ports[pattern_key] = set()
                    allowed_ports[pattern_key].add(entry.port)
            else:
                allowed_domains.add(entry.host)
                if entry.port is not None:
                    host_key = entry.host.lower()
                    if host_key not in allowed_ports:
                        allowed_ports[host_key] = set()
                    allowed_ports[host_key].add(entry.port)
        elif entry.kind in ("ip", "cidr"):
            # Store the host (IP/CIDR) without the port for matching.
            allowed_ips.add(entry.host)
            if entry.port is not None:
                host_key = entry.host.lower()
                if host_key not in allowed_ports:
                    allowed_ports[host_key] = set()
                allowed_ports[host_key].add(entry.port)
    return allowed_domains, allowed_wildcards, allowed_ips, allowed_ports


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from sandbox.network_allowlist import NetworkAllowList

    parser = argparse.ArgumentParser(description="SNI-aware egress proxy")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--allowlist", required=True,
                        help="Comma-separated allow-list (same format as --network-allowlist)")
    parser.add_argument("--allowlist-file", help="File with allow-list entries")
    parser.add_argument("--dns-resolver", default="", help="Restrict DNS to this resolver IP")
    parser.add_argument("--dns-audit", action="store_true",
                        help="Log every DNS resolution as a JSON line to stderr (audit trail)")
    parser.add_argument("--dns-rate-limit", type=int, default=None,
                        help="Max DNS queries per domain per window (default: unlimited)")
    parser.add_argument("--dns-rate-window", type=float, default=60.0,
                        help="Sliding window (seconds) for the per-domain DNS rate limit")
    args = parser.parse_args()

    # Parse the allow-list.
    if args.allowlist_file:
        al = NetworkAllowList.from_file(args.allowlist_file)
    else:
        al = NetworkAllowList.from_string(args.allowlist)

    # Extract domains and wildcards from the allow-list entries.
    # Also extract port restrictions: when an entry has a port (e.g.
    # pypi.org:443), only that port is allowed for that host.
    allowed_domains, allowed_wildcards, allowed_ips, allowed_ports = (
        extract_allowlist_sets(al)
    )

    proxy = SNIEgressProxy(
        allowed_domains=allowed_domains,
        allowed_wildcards=allowed_wildcards,
        allowed_ips=allowed_ips,
        allowed_ports=allowed_ports,
        port=args.port,
        host=args.host,
        dns_resolver=args.dns_resolver or None,
        dns_audit=args.dns_audit,
        dns_rate_limit=args.dns_rate_limit,
        dns_rate_window=args.dns_rate_window,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(proxy.start())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(proxy.stop())


if __name__ == "__main__":
    main()
