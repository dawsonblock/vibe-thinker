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
  3. For HTTPS: the proxy reads the SNI from the TLS ClientHello and
     checks it against the allow-list. If allowed, it tunnels the
     connection to the resolved IP (CONNECT proxy). If denied, it
     closes the connection.
  4. For HTTP: the proxy reads the Host header and checks it against
     the allow-list.

This is a transparent CONNECT proxy — it does NOT perform TLS
interception (no MITM). It only inspects the SNI (which is in
cleartext in the TLS ClientHello) and the Host header (which is in
cleartext for HTTP). The actual TLS traffic passes through untouched.

Usage:
    python3 -m sandbox.sni_proxy --port 8888 \\
        --allowlist "pypi.org:443,*.pypi.org:443,files.pythonhosted.org:443"

Then configure the sandbox to use it:
    --proxy-egress 127.0.0.1:8888
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# SNI extraction
# ---------------------------------------------------------------------------

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


def is_domain_allowed(
    hostname: str,
    allowed_domains: Set[str],
    allowed_wildcards: Set[str],
) -> bool:
    """Check if a hostname is allowed by the allow-list."""
    hostname = hostname.lower().strip()
    if hostname in allowed_domains:
        return True
    for pattern in allowed_wildcards:
        if domain_matches(hostname, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# SNI proxy server
# ---------------------------------------------------------------------------

class SNIEgressProxy:
    """Async SNI-aware egress proxy.

    Listens for CONNECT requests (HTTPS) and plain HTTP requests.
    For CONNECT: reads the SNI from the TLS ClientHello and checks it
    against the allow-list. For HTTP: reads the Host header.

    Allowed connections are tunneled to the destination. Denied
    connections are closed immediately.
    """

    def __init__(
        self,
        allowed_domains: Set[str],
        allowed_wildcards: Set[str],
        allowed_ips: Set[str] = None,
        port: int = 8888,
        host: str = "127.0.0.1",
        dns_resolver: Optional[str] = None,
    ):
        self.allowed_domains = allowed_domains
        self.allowed_wildcards = allowed_wildcards
        self.allowed_ips = allowed_ips or set()
        self.port = port
        self.host = host
        self.dns_resolver = dns_resolver
        self._server: Optional[asyncio.AbstractServer] = None
        self._denied_count = 0
        self._allowed_count = 0

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
        <TLS ClientHello with SNI>
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

        # Read and discard headers until empty line.
        while True:
            header = await asyncio.wait_for(
                client_reader.readline(), timeout=5.0
            )
            if header in (b"\r\n", b"\n", b""):
                break

        # Read the TLS ClientHello to extract SNI.
        try:
            client_hello = await asyncio.wait_for(
                client_reader.read(4096), timeout=5.0
            )
        except asyncio.TimeoutError:
            client_writer.close()
            self._denied_count += 1
            return

        sni = extract_sni(client_hello)
        if sni and is_domain_allowed(sni, self.allowed_domains, self.allowed_wildcards):
            # Allowed — tunnel the connection.
            self._allowed_count += 1
            try:
                # Send 200 Connection Established to the client.
                client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await client_writer.drain()
                # Forward the ClientHello we already read.
                client_writer.write(client_hello)
                await client_writer.drain()
                # Connect to the real destination.
                remote_reader, remote_writer = await asyncio.open_connection(
                    host, int(port)
                )
                remote_writer.write(client_hello)
                await remote_writer.drain()
                # Tunnel bidirectionally.
                await self._tunnel(client_reader, client_writer,
                                   remote_reader, remote_writer)
            except (ConnectionError, OSError, asyncio.TimeoutError):
                pass
        else:
            # Denied — SNI not in allow-list or no SNI.
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()

    async def _handle_http(
        self,
        first_line: str,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ):
        """Handle a plain HTTP request.

        Extract the Host header and check it against the allow-list.
        """
        # Read headers to find Host.
        headers = first_line + "\r\n"
        host = None
        while True:
            header = await asyncio.wait_for(
                client_reader.readline(), timeout=5.0
            )
            if header in (b"\r\n", b"\n", b""):
                break
            header_str = header.decode("ascii", errors="replace")
            headers += header_str
            if header_str.lower().startswith("host:"):
                host = header_str.split(":", 1)[1].strip()
                # Strip port from host.
                host = host.split(":")[0]

        if host and is_domain_allowed(host, self.allowed_domains, self.allowed_wildcards):
            self._allowed_count += 1
            # Forward the request to the destination.
            # Parse the URL from the request line.
            parts = first_line.split()
            url = parts[1] if len(parts) > 1 else "/"
            if url.startswith("http://"):
                # Absolute URL — extract host and path.
                url_parts = url[7:].split("/", 1)
                target_host = url_parts[0]
                path = "/" + url_parts[1] if len(url_parts) > 1 else "/"
            else:
                target_host = host
                path = url
            host_part, _, port_part = target_host.rpartition(":")
            if not host_part:
                host_part = target_host
                port_part = "80"
            try:
                remote_reader, remote_writer = await asyncio.open_connection(
                    host_part, int(port_part)
                )
                # Forward the request.
                remote_writer.write(f"{parts[0]} {path} HTTP/1.1\r\n".encode())
                remote_writer.write(headers.encode())
                await remote_writer.drain()
                # Tunnel bidirectionally.
                await self._tunnel(client_reader, client_writer,
                                   remote_reader, remote_writer)
            except (ConnectionError, OSError, asyncio.TimeoutError):
                pass
        else:
            self._denied_count += 1
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()

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
    args = parser.parse_args()

    # Parse the allow-list.
    if args.allowlist_file:
        al = NetworkAllowList.from_file(args.allowlist_file)
    else:
        al = NetworkAllowList.from_string(args.allowlist)

    # Extract domains and wildcards from the allow-list entries.
    allowed_domains: Set[str] = set()
    allowed_wildcards: Set[str] = set()
    allowed_ips: Set[str] = set()
    for entry in al.entries:
        if entry.kind == "domain":
            if entry.host.startswith("*."):
                allowed_wildcards.add(entry.host)
            else:
                allowed_domains.add(entry.host)
        elif entry.kind in ("ip", "cidr"):
            allowed_ips.add(entry.raw)

    proxy = SNIEgressProxy(
        allowed_domains=allowed_domains,
        allowed_wildcards=allowed_wildcards,
        allowed_ips=allowed_ips,
        port=args.port,
        host=args.host,
        dns_resolver=args.dns_resolver or None,
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
