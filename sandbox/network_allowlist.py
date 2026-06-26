"""Network allow-list for granular sandbox egress filtering.

Parses a list of allowed domains, IPs, and CIDR ranges, and generates
iptables rules that whitelist only those destinations. The Docker
executor applies these rules inside the container before running the
candidate code, replacing the binary --network=none / --network=default
choice with fine-grained egress control.

Trust model (fail-closed):
  - No allow-list -> --network=none (unchanged behavior, no egress)
  - Empty allow-list -> --network=none (deny all, same as no list)
  - Allow-list with entries -> --network=default + iptables rules that
    DROP all egress except the allow-listed destinations
  - DNS resolution for allow-listed domains happens at rule-generation
    time (on the host). If a domain doesn't resolve, it's skipped with
    a warning — the candidate code won't be able to reach it, but the
    sandbox still runs (fail-closed for that domain, not for the whole
    execution).

Supported allow-list entry formats:
  - Domain: ``pypi.org`` or ``*.pypi.org`` (wildcard subdomain)
  - IPv4: ``10.0.0.1``
  - CIDR: ``10.0.0.0/24``
  - Port-specific: ``pypi.org:443`` or ``10.0.0.0/24:5432``

The iptables rules:
  1. Allow loopback (lo interface)
  2. Allow established/related connections (return traffic)
  3. Allow DNS (UDP/TCP port 53) to any resolver — needed for the
     candidate code to resolve allow-listed domains at runtime
  4. For each allow-listed destination: allow OUTPUT to that IP/CIDR
     (optionally restricted to a specific port)
  5. DROP everything else

Note: DNS is allowed to any resolver because the iptables rules use
IP addresses (resolved at rule-generation time). If we blocked DNS,
the candidate code couldn't resolve any domain, including allow-listed
ones. The alternative (hardcoding resolved IPs) is already done in the
rules — DNS is only needed if the candidate code itself does hostname
resolution at runtime.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AllowListEntry:
    """A single entry in the network allow-list.

    Attributes:
        raw: the original string (e.g. "pypi.org:443")
        kind: "domain", "ip", "cidr"
        host: the domain, IP, or CIDR (without port)
        port: optional port restriction (None = all ports)
        wildcard: True for "*.example.com" domain patterns
    """
    raw: str
    kind: str  # "domain", "ip", "cidr"
    host: str
    port: Optional[int] = None
    wildcard: bool = False

    def resolved_ips(self) -> List[str]:
        """Resolve this entry to a list of IP addresses (for iptables rules).

        Domains are resolved via DNS at call time. IPs and CIDRs return
        themselves. Wildcard domains (``*.example.com``) cannot be
        resolved directly — they're matched at the DNS layer, so we
        return [] and rely on the DNS allow rule. The caller should
        handle wildcards separately (by allowing DNS and then filtering
        at the application layer, or by resolving known subdomains).

        Returns:
            List of IP address strings. Empty for wildcard domains or
            unresolvable domains.
        """
        if self.kind == "ip" or self.kind == "cidr":
            return [self.host]
        if self.kind == "domain" and not self.wildcard:
            try:
                infos = socket.getaddrinfo(self.host, None, socket.AF_INET)
                return list({info[4][0] for info in infos})
            except socket.gaierror:
                return []
        return []


class NetworkAllowList:
    """Parses and manages a network egress allow-list.

    Usage:
        allowlist = NetworkAllowList.from_string("pypi.org:443,10.0.0.0/24")
        rules = allowlist.generate_iptables_rules()
        # rules is a list of shell commands to run inside the container

    Or from a file:
        allowlist = NetworkAllowList.from_file("allowlist.txt")
    """

    def __init__(self, entries: List[AllowListEntry]):
        self.entries = entries

    @classmethod
    def from_string(cls, spec: str) -> "NetworkAllowList":
        """Parse a comma-separated allow-list string.

        Entries are separated by commas. Whitespace is stripped.
        Empty entries are ignored. Lines starting with # are comments
        (when parsed from a file).
        """
        entries = []
        for part in spec.split(","):
            part = part.strip()
            if not part or part.startswith("#"):
                continue
            entry = cls._parse_entry(part)
            if entry is not None:
                entries.append(entry)
        return cls(entries)

    @classmethod
    def from_file(cls, path: str) -> "NetworkAllowList":
        """Parse an allow-list from a file (one entry per line)."""
        with open(path, "r") as f:
            lines = f.readlines()
        entries = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = cls._parse_entry(line)
            if entry is not None:
                entries.append(entry)
        return cls(entries)

    @staticmethod
    def _parse_entry(raw: str) -> Optional[AllowListEntry]:
        """Parse a single allow-list entry.

        Formats:
          - domain: pypi.org
          - wildcard: *.pypi.org
          - IPv4: 10.0.0.1
          - CIDR: 10.0.0.0/24
          - With port: pypi.org:443, 10.0.0.0/24:5432
        """
        port: Optional[int] = None
        host_part = raw

        # Extract port if present (last :N).
        # But be careful: IPv6 addresses contain colons. For now, only
        # support IPv4 (the sandbox uses python:3.12-slim which is IPv4).
        if ":" in raw:
            # Check if it's a CIDR (has /) before the colon.
            cidr_match = re.match(r"^(\d+\.\d+\.\d+\.\d+/\d+):(\d+)$", raw)
            if cidr_match:
                host_part = cidr_match.group(1)
                port = int(cidr_match.group(2))
            else:
                # Try domain:port or ip:port
                parts = raw.rsplit(":", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    host_part = parts[0]
                    port = int(parts[1])

        # Determine the kind.
        # CIDR?
        if "/" in host_part:
            try:
                ipaddress.ip_network(host_part, strict=False)
                return AllowListEntry(
                    raw=raw, kind="cidr", host=host_part, port=port,
                )
            except ValueError:
                pass

        # IP address?
        try:
            ipaddress.ip_address(host_part)
            return AllowListEntry(
                raw=raw, kind="ip", host=host_part, port=port,
            )
        except ValueError:
            pass

        # Domain (including wildcards)?
        if host_part.startswith("*."):
            domain = host_part[2:]
            if NetworkAllowList._is_valid_domain(domain):
                return AllowListEntry(
                    raw=raw, kind="domain", host=domain,
                    port=port, wildcard=True,
                )
        elif NetworkAllowList._is_valid_domain(host_part):
            return AllowListEntry(
                raw=raw, kind="domain", host=host_part, port=port,
            )

        # Unparseable entry — skip it (fail-closed: don't allow unknowns).
        print(f"[NetworkAllowList] Warning: could not parse entry "
              f"{raw!r} — skipping (fail-closed)")
        return None

    @staticmethod
    def _is_valid_domain(domain: str) -> bool:
        """Basic domain validation."""
        if not domain or len(domain) > 253:
            return False
        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*$", domain):
            return False
        return True

    @property
    def is_empty(self) -> bool:
        """True if the allow-list has no entries (deny all egress)."""
        return len(self.entries) == 0

    def generate_iptables_rules(
        self, dns_resolver: Optional[str] = None,
    ) -> List[str]:
        """Generate iptables shell commands to enforce the allow-list.

        Returns a list of shell command strings to run inside the
        container. The commands:
          1. Allow loopback
          2. Allow established/related connections
          3. Allow DNS (port 53) — to a specific resolver if
             `dns_resolver` is set, otherwise to any resolver
          4. For each allow-listed IP/CIDR: allow OUTPUT
          5. Drop everything else (IPv4)
          6. Deny all IPv6 (ip6tables DROP policy)

        Domain entries are resolved to IPs at this point. Wildcard
        domains are skipped (they rely on the DNS allow rule + the
        candidate code's own hostname resolution).

        The commands use `iptables` and `ip6tables` which must be
        available in the container image. The purpose-built
        `vibe-thinker-sandbox` image includes them baked in — no
        apt-get at runtime (see sandbox/Dockerfile).

        Args:
            dns_resolver: optional IP address of a DNS resolver to
                restrict DNS queries to. When None, DNS is allowed to
                any resolver (needed for hostname resolution). When set,
                only the specified resolver can receive DNS queries,
                preventing DNS-based data exfiltration through arbitrary
                resolvers.
        """
        if self.is_empty:
            # No entries -> deny all (equivalent to --network=none,
            # but we still allow loopback for the script to function).
            # Also deny all IPv6.
            return [
                "iptables -P OUTPUT DROP",
                "iptables -A OUTPUT -o lo -j ACCEPT",
                "iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT",
                "ip6tables -P OUTPUT DROP",
                "ip6tables -A OUTPUT -o lo -j ACCEPT",
                "ip6tables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT",
            ]

        rules: List[str] = []

        # 1. Allow loopback.
        rules.append("iptables -A OUTPUT -o lo -j ACCEPT")

        # 2. Allow established/related connections (return traffic).
        rules.append(
            "iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT"
        )

        # 3. Allow DNS (UDP/TCP port 53).
        #    If a specific resolver is provided, restrict DNS to that
        #    resolver only. Otherwise, allow DNS to any resolver (needed
        #    because iptables rules use IPs, but the candidate code may
        #    need to resolve hostnames at runtime).
        if dns_resolver:
            rules.append(
                f"iptables -A OUTPUT -d {dns_resolver} -p udp --dport 53 -j ACCEPT"
            )
            rules.append(
                f"iptables -A OUTPUT -d {dns_resolver} -p tcp --dport 53 -j ACCEPT"
            )
        else:
            rules.append("iptables -A OUTPUT -p udp --dport 53 -j ACCEPT")
            rules.append("iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT")

        # 4. Allow each allow-listed destination.
        resolved_count = 0
        for entry in self.entries:
            if entry.wildcard:
                # Wildcard domains can't be resolved to IPs. They rely
                # on the DNS allow rule + the candidate code resolving
                # the hostname at runtime. The actual IP filtering is
                # not possible for wildcards without a DNS proxy.
                print(f"[NetworkAllowList] Wildcard domain "
                      f"*.{entry.host} — relies on DNS allow rule "
                      f"(no IP-level filtering for this entry)")
                continue

            ips = entry.resolved_ips()
            if not ips:
                print(f"[NetworkAllowList] Could not resolve {entry.host} "
                      f"— skipping (fail-closed for this domain)")
                continue

            for ip in ips:
                port_spec = f"--dport {entry.port}" if entry.port else ""
                if port_spec:
                    rules.append(
                        f"iptables -A OUTPUT -d {ip} -p tcp {port_spec} -j ACCEPT"
                    )
                    rules.append(
                        f"iptables -A OUTPUT -d {ip} -p udp {port_spec} -j ACCEPT"
                    )
                else:
                    rules.append(
                        f"iptables -A OUTPUT -d {ip} -j ACCEPT"
                    )
                resolved_count += 1

        # 5. Drop everything else (IPv4).
        rules.append("iptables -P OUTPUT DROP")

        # 6. Deny all IPv6 egress.
        #    The allow-list rules above only cover IPv4 (iptables). To
        #    prevent IPv6 bypass, we set ip6tables OUTPUT policy to DROP
        #    and allow only loopback + established connections. If IPv6
        #    egress is needed, it should be explicitly allow-listed.
        rules.append("ip6tables -P OUTPUT DROP")
        rules.append("ip6tables -A OUTPUT -o lo -j ACCEPT")
        rules.append(
            "ip6tables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT"
        )

        return rules

    def generate_docker_run_args(self) -> List[str]:
        """Generate Docker run arguments for the allow-list.

        When an allow-list is active, we use --network=default (bridge)
        instead of --network=none, and apply iptables rules inside the
        container. Returns the Docker --network argument to use.

        Returns:
            ["--network", "default"] if the allow-list has entries,
            ["--network", "none"] if empty (deny all, no bridge needed).
        """
        if self.is_empty:
            return ["--network", "none"]
        return ["--network", "default"]

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for logging/debugging."""
        return {
            "entry_count": len(self.entries),
            "domains": [e.raw for e in self.entries
                        if e.kind == "domain" and not e.wildcard],
            "wildcards": [e.raw for e in self.entries if e.wildcard],
            "ips": [e.raw for e in self.entries if e.kind == "ip"],
            "cidrs": [e.raw for e in self.entries if e.kind == "cidr"],
            "is_empty": self.is_empty,
        }
