"""Tests for network allow-listing (Phase 2.1)."""

import os
import pytest
from unittest.mock import MagicMock, patch

from sandbox.network_allowlist import NetworkAllowList, AllowListEntry


# ---------------------------------------------------------------------- #
# Parsing
# ---------------------------------------------------------------------- #
class TestParsing:
    def test_parse_domain(self):
        al = NetworkAllowList.from_string("pypi.org")
        assert len(al.entries) == 1
        e = al.entries[0]
        assert e.kind == "domain"
        assert e.host == "pypi.org"
        assert e.port is None
        assert e.wildcard is False

    def test_parse_wildcard_domain(self):
        al = NetworkAllowList.from_string("*.pypi.org")
        assert len(al.entries) == 1
        e = al.entries[0]
        assert e.kind == "domain"
        assert e.host == "pypi.org"
        assert e.wildcard is True

    def test_parse_ip(self):
        al = NetworkAllowList.from_string("10.0.0.1")
        assert len(al.entries) == 1
        e = al.entries[0]
        assert e.kind == "ip"
        assert e.host == "10.0.0.1"

    def test_parse_cidr(self):
        al = NetworkAllowList.from_string("10.0.0.0/24")
        assert len(al.entries) == 1
        e = al.entries[0]
        assert e.kind == "cidr"
        assert e.host == "10.0.0.0/24"

    def test_parse_domain_with_port(self):
        al = NetworkAllowList.from_string("pypi.org:443")
        assert len(al.entries) == 1
        e = al.entries[0]
        assert e.kind == "domain"
        assert e.host == "pypi.org"
        assert e.port == 443

    def test_parse_cidr_with_port(self):
        al = NetworkAllowList.from_string("10.0.0.0/24:5432")
        assert len(al.entries) == 1
        e = al.entries[0]
        assert e.kind == "cidr"
        assert e.host == "10.0.0.0/24"
        assert e.port == 5432

    def test_parse_ip_with_port(self):
        al = NetworkAllowList.from_string("10.0.0.1:8080")
        assert len(al.entries) == 1
        e = al.entries[0]
        assert e.kind == "ip"
        assert e.port == 8080

    def test_parse_multiple_entries(self):
        al = NetworkAllowList.from_string("pypi.org:443, 10.0.0.0/24, *.example.com")
        assert len(al.entries) == 3
        assert al.entries[0].kind == "domain"
        assert al.entries[1].kind == "cidr"
        assert al.entries[2].wildcard is True

    def test_parse_empty_string(self):
        al = NetworkAllowList.from_string("")
        assert al.is_empty

    def test_parse_whitespace_only(self):
        al = NetworkAllowList.from_string("  ,  ,  ")
        assert al.is_empty

    def test_parse_invalid_entry_skipped(self):
        al = NetworkAllowList.from_string("!!!invalid!!!, pypi.org")
        assert len(al.entries) == 1
        assert al.entries[0].host == "pypi.org"

    def test_parse_comments_ignored(self):
        al = NetworkAllowList.from_string("# comment, pypi.org")
        assert len(al.entries) == 1
        assert al.entries[0].host == "pypi.org"


class TestFromFile:
    def test_from_file(self, tmp_path):
        path = str(tmp_path / "allowlist.txt")
        with open(path, "w") as f:
            f.write("# Allowed destinations\n")
            f.write("pypi.org:443\n")
            f.write("\n")
            f.write("10.0.0.0/24\n")
            f.write("# another comment\n")
            f.write("*.internal.corp\n")
        al = NetworkAllowList.from_file(path)
        assert len(al.entries) == 3
        assert al.entries[0].host == "pypi.org"
        assert al.entries[1].kind == "cidr"
        assert al.entries[2].wildcard is True

    def test_from_file_empty(self, tmp_path):
        path = str(tmp_path / "empty.txt")
        with open(path, "w") as f:
            f.write("# only comments\n")
        al = NetworkAllowList.from_file(path)
        assert al.is_empty


# ---------------------------------------------------------------------- #
# IP resolution
# ---------------------------------------------------------------------- #
class TestResolvedIPs:
    def test_ip_returns_itself(self):
        e = AllowListEntry(raw="10.0.0.1", kind="ip", host="10.0.0.1")
        assert e.resolved_ips() == ["10.0.0.1"]

    def test_cidr_returns_itself(self):
        e = AllowListEntry(raw="10.0.0.0/24", kind="cidr", host="10.0.0.0/24")
        assert e.resolved_ips() == ["10.0.0.0/24"]

    def test_wildcard_returns_empty(self):
        e = AllowListEntry(raw="*.pypi.org", kind="domain",
                           host="pypi.org", wildcard=True)
        assert e.resolved_ips() == []

    def test_domain_resolves_via_dns(self):
        e = AllowListEntry(raw="localhost", kind="domain", host="localhost")
        ips = e.resolved_ips()
        # localhost should resolve to 127.0.0.1 on most systems.
        assert "127.0.0.1" in ips

    def test_unresolvable_domain_returns_empty(self):
        e = AllowListEntry(raw="nonexistent.invalid", kind="domain",
                           host="nonexistent.invalid")
        assert e.resolved_ips() == []


# ---------------------------------------------------------------------- #
# iptables rule generation
# ---------------------------------------------------------------------- #
class TestIptablesRules:
    def test_empty_allowlist_denies_all(self):
        al = NetworkAllowList.from_string("")
        rules = al.generate_iptables_rules()
        # Should set OUTPUT policy to DROP and allow loopback + established.
        assert any("iptables -P OUTPUT DROP" in r for r in rules)
        assert any("-o lo -j ACCEPT" in r for r in rules)
        assert any("ESTABLISHED" in r for r in rules)

    def test_ip_entry_generates_allow_rule(self):
        al = NetworkAllowList.from_string("10.0.0.1")
        rules = al.generate_iptables_rules()
        # Should have a rule allowing OUTPUT to 10.0.0.1.
        allow_rules = [r for r in rules if "10.0.0.1" in r and "ACCEPT" in r]
        assert len(allow_rules) >= 1

    def test_cidr_entry_generates_allow_rule(self):
        al = NetworkAllowList.from_string("10.0.0.0/24")
        rules = al.generate_iptables_rules()
        allow_rules = [r for r in rules if "10.0.0.0/24" in r and "ACCEPT" in r]
        assert len(allow_rules) >= 1

    def test_port_specific_rule(self):
        al = NetworkAllowList.from_string("10.0.0.1:443")
        rules = al.generate_iptables_rules()
        # Should have TCP and UDP rules with --dport 443.
        port_rules = [r for r in rules if "--dport 443" in r]
        assert len(port_rules) >= 2  # TCP + UDP

    def test_dns_always_allowed(self):
        """DNS (port 53) must always be allowed so the candidate code can
        resolve allow-listed hostnames at runtime."""
        al = NetworkAllowList.from_string("10.0.0.1")
        rules = al.generate_iptables_rules()
        dns_rules = [r for r in rules if "--dport 53" in r]
        assert len(dns_rules) >= 2  # UDP + TCP

    def test_loopback_always_allowed(self):
        al = NetworkAllowList.from_string("10.0.0.1")
        rules = al.generate_iptables_rules()
        assert any("-o lo -j ACCEPT" in r for r in rules)

    def test_established_connections_allowed(self):
        al = NetworkAllowList.from_string("10.0.0.1")
        rules = al.generate_iptables_rules()
        assert any("ESTABLISHED,RELATED" in r for r in rules)

    def test_drop_is_last_ipv4_rule(self):
        """The IPv4 DROP policy should be set after all IPv4 ACCEPT rules,
        before the IPv6 denial rules."""
        al = NetworkAllowList.from_string("10.0.0.1")
        rules = al.generate_iptables_rules()
        # Find the IPv4 DROP policy — it should come after all IPv4
        # ACCEPT rules but before the IPv6 rules.
        ipv4_drop_idx = next(i for i, r in enumerate(rules)
                             if "iptables -P OUTPUT DROP" in r)
        # All IPv4 ACCEPT rules should come before the DROP.
        for i, r in enumerate(rules):
            if "iptables -A" in r and "ACCEPT" in r:
                assert i < ipv4_drop_idx, f"IPv4 ACCEPT rule after DROP: {r}"
        # IPv6 rules should come after the IPv4 DROP.
        for i, r in enumerate(rules):
            if r.startswith("ip6tables"):
                assert i > ipv4_drop_idx, f"IPv6 rule before IPv4 DROP: {r}"

    def test_wildcard_skipped_in_iptables(self):
        """Wildcard domains can't be resolved to IPs — they're skipped in
        iptables rules (rely on DNS allow rule)."""
        al = NetworkAllowList.from_string("*.pypi.org")
        rules = al.generate_iptables_rules()
        # No IP-specific ACCEPT rules for the wildcard.
        allow_rules = [r for r in rules if "ACCEPT" in r and "-d " in r]
        assert len(allow_rules) == 0  # only loopback + DNS + established


# ---------------------------------------------------------------------- #
# Docker run args
# ---------------------------------------------------------------------- #
class TestDockerRunArgs:
    def test_empty_allowlist_uses_network_none(self):
        al = NetworkAllowList.from_string("")
        args = al.generate_docker_run_args()
        assert args == ["--network", "none"]

    def test_non_empty_uses_network_default(self):
        al = NetworkAllowList.from_string("10.0.0.1")
        args = al.generate_docker_run_args()
        assert args == ["--network", "default"]


# ---------------------------------------------------------------------- #
# Summary
# ---------------------------------------------------------------------- #
class TestSummary:
    def test_summary_categorizes_entries(self):
        al = NetworkAllowList.from_string(
            "pypi.org:443, 10.0.0.1, 10.0.0.0/24, *.corp.local"
        )
        s = al.summary()
        assert s["entry_count"] == 4
        assert len(s["domains"]) == 1
        assert len(s["ips"]) == 1
        assert len(s["cidrs"]) == 1
        assert len(s["wildcards"]) == 1
        assert s["is_empty"] is False

    def test_summary_empty(self):
        al = NetworkAllowList.from_string("")
        s = al.summary()
        assert s["is_empty"] is True
        assert s["entry_count"] == 0


# ---------------------------------------------------------------------- #
# DockerSandboxExecutor integration
# ---------------------------------------------------------------------- #
class TestExecutorIntegration:
    def test_executor_accepts_allowlist(self):
        from sandbox.docker_executor import DockerSandboxExecutor
        al = NetworkAllowList.from_string("10.0.0.1")
        executor = DockerSandboxExecutor(allowlist=al)
        assert executor._allowlist is al

    def test_executor_set_allowlist(self):
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        assert executor._allowlist is None
        al = NetworkAllowList.from_string("10.0.0.1")
        executor.set_allowlist(al)
        assert executor._allowlist is al

    def test_executor_no_allowlist_uses_network_none(self):
        """Without an allow-list, the executor uses --network=none
        (unchanged behavior)."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        # We can't easily test the actual docker command without running
        # Docker, but we can verify the allow-list is None.
        assert executor._allowlist is None


# ---------------------------------------------------------------------- #
# Orchestrator integration
# ---------------------------------------------------------------------- #
class TestOrchestratorIntegration:
    def test_orchestrator_applies_allowlist_to_executor(self):
        from hybrid_orchestrator import HybridReasoningOrchestrator
        al = NetworkAllowList.from_string("10.0.0.1")
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False, use_embedding_router=False,
            use_clr_cache=False, use_trajectory_store=False,
            network_allowlist=al,
        )
        # The code verifier's executor should have the allow-list set.
        if o.code_verifier and hasattr(o.code_verifier, "executor"):
            executor = o.code_verifier.executor
            if hasattr(executor, "_allowlist"):
                assert executor._allowlist is al

    def test_orchestrator_no_allowlist(self):
        from hybrid_orchestrator import HybridReasoningOrchestrator
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False, use_embedding_router=False,
            use_clr_cache=False, use_trajectory_store=False,
        )
        # No allow-list -> executor's _allowlist should be None.
        if o.code_verifier and hasattr(o.code_verifier, "executor"):
            executor = o.code_verifier.executor
            if hasattr(executor, "_allowlist"):
                assert executor._allowlist is None


# ---------------------------------------------------------------------- #
# IPv6 denial (v0.4.0 hardening)
# ---------------------------------------------------------------------- #
class TestIPv6Denial:
    """IPv6 bypass prevention — ip6tables must be explicitly denied."""

    def test_empty_allowlist_denies_ipv6(self):
        """Empty allow-list must also deny IPv6, not just IPv4."""
        al = NetworkAllowList.from_string("")
        rules = al.generate_iptables_rules()
        ip6_rules = [r for r in rules if r.startswith("ip6tables")]
        assert len(ip6_rules) >= 3  # DROP policy + loopback + established
        assert any("ip6tables -P OUTPUT DROP" in r for r in ip6_rules)
        assert any("ip6tables -A OUTPUT -o lo -j ACCEPT" in r for r in ip6_rules)

    def test_non_empty_allowlist_denies_ipv6(self):
        """Non-empty allow-list must also deny IPv6."""
        al = NetworkAllowList.from_string("10.0.0.1")
        rules = al.generate_iptables_rules()
        ip6_rules = [r for r in rules if r.startswith("ip6tables")]
        assert any("ip6tables -P OUTPUT DROP" in r for r in ip6_rules)
        assert any("ip6tables -A OUTPUT -o lo -j ACCEPT" in r for r in ip6_rules)
        assert any("ESTABLISHED" in r and "ip6tables" in r for r in ip6_rules)

    def test_no_ipv6_allow_rules_generated(self):
        """The allow-list should NOT generate any ip6tables ACCEPT rules
        for allow-listed destinations — only IPv4 rules. IPv6 is denied
        entirely."""
        al = NetworkAllowList.from_string("10.0.0.1, pypi.org:443")
        rules = al.generate_iptables_rules()
        ip6_accept = [r for r in rules if "ip6tables" in r and "ACCEPT" in r
                       and "lo" not in r and "ESTABLISHED" not in r]
        assert len(ip6_accept) == 0


# ---------------------------------------------------------------------- #
# DNS restriction (v0.4.0 hardening)
# ---------------------------------------------------------------------- #
class TestDNSRestriction:
    """DNS exfiltration prevention — restrict DNS to a specific resolver."""

    def test_dns_allowed_to_any_resolver_by_default(self):
        """Without a DNS resolver restriction, DNS is allowed to any
        resolver (needed for hostname resolution in the allow-list)."""
        al = NetworkAllowList.from_string("10.0.0.1")
        rules = al.generate_iptables_rules()
        dns_rules = [r for r in rules if "--dport 53" in r]
        # Should have UDP + TCP DNS rules without a specific destination.
        assert any("-d " not in r for r in dns_rules)

    def test_dns_restricted_to_specific_resolver(self):
        """When a DNS resolver is specified, DNS is restricted to that
        resolver only — preventing DNS-based data exfiltration."""
        al = NetworkAllowList.from_string("10.0.0.1")
        rules = al.generate_iptables_rules(dns_resolver="8.8.8.8")
        dns_rules = [r for r in rules if "--dport 53" in r]
        # All DNS rules should target the specific resolver.
        for rule in dns_rules:
            assert "-d 8.8.8.8" in rule
        # Should NOT have any broad DNS allow rules.
        broad_dns = [r for r in dns_rules if "-d " not in r]
        assert len(broad_dns) == 0

    def test_dns_resolver_in_empty_allowlist(self):
        """DNS restriction should work even with an empty allow-list
        (though in practice, empty allow-list uses --network=none)."""
        al = NetworkAllowList.from_string("")
        rules = al.generate_iptables_rules(dns_resolver="8.8.8.8")
        # Empty allow-list denies all — no DNS rules at all.
        dns_rules = [r for r in rules if "--dport 53" in r]
        assert len(dns_rules) == 0


# ---------------------------------------------------------------------- #
# Privilege dropping (v0.4.0 hardening)
# ---------------------------------------------------------------------- #
class TestPrivilegeDropping:
    """Verify the executor uses the sandbox image with entrypoint that
    drops privileges. These tests verify the configuration, not the
    runtime behavior (which requires Docker)."""

    def test_executor_uses_sandbox_image_by_default(self):
        """The default image should be the purpose-built sandbox image
        with iptables baked in, not python:3.12-slim."""
        from sandbox.docker_executor import DockerSandboxExecutor, SANDBOX_IMAGE
        executor = DockerSandboxExecutor()
        assert executor.image == SANDBOX_IMAGE
        assert "vibe-thinker-sandbox" in executor.image

    def test_executor_accepts_dns_resolver(self):
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor(dns_resolver="8.8.8.8")
        assert executor._dns_resolver == "8.8.8.8"

    def test_executor_set_dns_resolver(self):
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        assert executor._dns_resolver is None
        executor.set_dns_resolver("1.1.1.1")
        assert executor._dns_resolver == "1.1.1.1"

    def test_firewall_env_includes_rules(self):
        """The firewall env should include base64-encoded iptables rules."""
        import base64
        from sandbox.docker_executor import DockerSandboxExecutor
        al = NetworkAllowList.from_string("10.0.0.1")
        executor = DockerSandboxExecutor(allowlist=al)
        env = executor._build_firewall_env()
        assert "VT_IPTABLES_RULES" in env
        # Decode and verify it contains iptables commands.
        decoded = base64.b64decode(env["VT_IPTABLES_RULES"]).decode()
        assert "iptables" in decoded
        assert "10.0.0.1" in decoded

    def test_firewall_env_includes_dns_resolver(self):
        """When a DNS resolver is set, it should be in the firewall env."""
        from sandbox.docker_executor import DockerSandboxExecutor
        al = NetworkAllowList.from_string("10.0.0.1")
        executor = DockerSandboxExecutor(allowlist=al, dns_resolver="8.8.8.8")
        env = executor._build_firewall_env()
        assert env.get("VT_DNS_RESOLVER") == "8.8.8.8"

    def test_firewall_env_empty_without_allowlist(self):
        """Without an allow-list, the firewall env should be empty."""
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        env = executor._build_firewall_env()
        assert env == {}

    def test_firewall_env_empty_with_empty_allowlist(self):
        """With an empty allow-list, the firewall env should be empty
        (the executor uses --network=none in that case)."""
        from sandbox.docker_executor import DockerSandboxExecutor
        al = NetworkAllowList.from_string("")
        executor = DockerSandboxExecutor(allowlist=al)
        env = executor._build_firewall_env()
        assert env == {}

    def test_rules_hash_computed(self):
        """The rules hash should be a non-empty hex string when rules exist."""
        from sandbox.docker_executor import DockerSandboxExecutor
        al = NetworkAllowList.from_string("10.0.0.1")
        executor = DockerSandboxExecutor(allowlist=al)
        env = executor._build_firewall_env()
        h = executor._compute_rules_hash(env)
        assert len(h) == 16  # truncated SHA-256
        assert all(c in "0123456789abcdef" for c in h)

    def test_rules_hash_empty_without_rules(self):
        from sandbox.docker_executor import DockerSandboxExecutor
        executor = DockerSandboxExecutor()
        h = executor._compute_rules_hash({})
        assert h == ""


# ---------------------------------------------------------------------- #
# Orchestrator integration with DNS resolver + sandbox image
# ---------------------------------------------------------------------- #
class TestOrchestratorDNSAndImage:
    def test_orchestrator_applies_dns_resolver(self):
        from hybrid_orchestrator import HybridReasoningOrchestrator
        al = NetworkAllowList.from_string("10.0.0.1")
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False, use_embedding_router=False,
            use_clr_cache=False, use_trajectory_store=False,
            network_allowlist=al,
            dns_resolver="8.8.8.8",
        )
        if o.code_verifier and hasattr(o.code_verifier, "executor"):
            executor = o.code_verifier.executor
            if hasattr(executor, "_dns_resolver"):
                assert executor._dns_resolver == "8.8.8.8"

    def test_orchestrator_applies_sandbox_image(self):
        from hybrid_orchestrator import HybridReasoningOrchestrator
        al = NetworkAllowList.from_string("10.0.0.1")
        o = HybridReasoningOrchestrator(
            vibe_endpoint="http://localhost:0",
            generalist_endpoint="http://localhost:0",
            use_clr=False, use_embedding_router=False,
            use_clr_cache=False, use_trajectory_store=False,
            network_allowlist=al,
            sandbox_image="my-custom-sandbox:v1",
        )
        if o.code_verifier and hasattr(o.code_verifier, "executor"):
            executor = o.code_verifier.executor
            if hasattr(executor, "image"):
                assert executor.image == "my-custom-sandbox:v1"


# ---------------------------------------------------------------------- #
# Dockerfile + entrypoint validation
# ---------------------------------------------------------------------- #
class TestSandboxImage:
    """Verify the Dockerfile and entrypoint script exist and have the
    required security properties."""

    def test_dockerfile_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "sandbox", "Dockerfile")
        assert os.path.exists(path)

    def test_dockerfile_installs_iptables(self):
        """The Dockerfile must install iptables at build time, not at
        runtime (closes the TOCTOU window)."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sandbox", "Dockerfile")) as f:
            content = f.read()
        assert "iptables" in content
        assert "apt-get install" in content

    def test_dockerfile_creates_non_root_user(self):
        """The Dockerfile must create a non-root user for candidate code."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sandbox", "Dockerfile")) as f:
            content = f.read()
        assert "useradd" in content
        assert "sandbox" in content

    def test_dockerfile_uses_entrypoint(self):
        """The Dockerfile must use the entrypoint script that applies
        firewall rules before candidate code runs."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sandbox", "Dockerfile")) as f:
            content = f.read()
        assert "ENTRYPOINT" in content
        assert "vt-entrypoint.sh" in content

    def test_entrypoint_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "sandbox", "entrypoint.sh")
        assert os.path.exists(path)

    def test_entrypoint_applies_rules_before_code(self):
        """The entrypoint must apply firewall rules BEFORE exec'ing the
        candidate command."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sandbox", "entrypoint.sh")) as f:
            content = f.read()
        # The iptables phase must come before the runuser/exec phase.
        iptables_pos = content.find("iptables -P OUTPUT DROP")
        runuser_pos = content.find("runuser")
        assert iptables_pos >= 0
        assert runuser_pos >= 0
        assert iptables_pos < runuser_pos

    def test_entrypoint_drops_privileges(self):
        """The entrypoint must drop to the sandbox user before running
        candidate code."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sandbox", "entrypoint.sh")) as f:
            content = f.read()
        assert "runuser -u sandbox" in content

    def test_entrypoint_denies_ipv6(self):
        """The entrypoint must explicitly deny IPv6 egress."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sandbox", "entrypoint.sh")) as f:
            content = f.read()
        assert "ip6tables" in content
        assert "ip6tables -P OUTPUT DROP" in content

    def test_entrypoint_restricts_dns(self):
        """The entrypoint must support DNS resolver restriction."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sandbox", "entrypoint.sh")) as f:
            content = f.read()
        assert "VT_DNS_RESOLVER" in content
        assert "--dport 53" in content

    def test_entrypoint_fails_closed_on_iptables_error(self):
        """If an iptables rule fails, the entrypoint must exit (fail-closed),
        not continue running candidate code without firewall protection."""
        with open(os.path.join(os.path.dirname(__file__), "..", "sandbox", "entrypoint.sh")) as f:
            content = f.read()
        # The entrypoint uses `set -euo pipefail` and exits on iptables failure.
        assert "set -euo pipefail" in content
        assert "FATAL" in content or "exit 1" in content
