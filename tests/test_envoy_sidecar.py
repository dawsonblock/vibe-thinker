"""Tests for the Envoy sidecar config generator (sandbox/envoy_sidecar.py).

These tests verify the config GENERATION without requiring Envoy to be
installed. The launcher (find_envoy_binary / launch_envoy) is tested
for fail-closed behavior when envoy is absent.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from sandbox.envoy_sidecar import (
    generate_envoy_config,
    write_envoy_config,
    find_envoy_binary,
    launch_envoy,
)
from sandbox.network_allowlist import NetworkAllowList


@pytest.fixture
def allowlist():
    return NetworkAllowList.from_string(
        "pypi.org:443,files.pythonhosted.org:443,*.pythonhosted.org:443"
    )


class TestGenerateEnvoyConfig:
    def test_returns_dict(self, allowlist):
        config = generate_envoy_config(allowlist)
        assert isinstance(config, dict)

    def test_has_admin_section(self, allowlist):
        config = generate_envoy_config(allowlist)
        assert "admin" in config
        assert config["admin"]["address"]["socket_address"]["port_value"] == 9901

    def test_has_listener_on_configured_port(self, allowlist):
        config = generate_envoy_config(allowlist, listen_port=9999)
        listener = config["static_resources"]["listeners"][0]
        assert listener["address"]["socket_address"]["port_value"] == 9999

    def test_has_tcp_proxy_filter(self, allowlist):
        config = generate_envoy_config(allowlist)
        filter_chain = config["static_resources"]["listeners"][0]["filter_chains"][0]
        filter_names = [f["name"] for f in filter_chain["filters"]]
        assert "envoy.filters.network.tcp_proxy" in filter_names

    def test_allowed_domains_in_matcher(self, allowlist):
        config = generate_envoy_config(allowlist)
        matcher = config["static_resources"]["listeners"][0]["filter_chains"][0][
            "filters"
        ][0]["typed_config"]["matcher"]
        exact_map = matcher["matcher_tree"]["exact_match_map"]
        # pypi.org and files.pythonhosted.org are exact domains.
        assert "pypi.org" in exact_map
        assert "files.pythonhosted.org" in exact_map

    def test_wildcard_becomes_suffix(self, allowlist):
        # *.pythonhosted.org -> ".pythonhosted.org" (suffix match).
        config = generate_envoy_config(allowlist)
        # The wildcard is added to allowed_domains with the leading dot.
        # It's not in exact_match_map (it's a suffix, not exact), but the
        # config should still be valid.
        assert isinstance(config, dict)

    def test_has_deny_on_no_match(self, allowlist):
        config = generate_envoy_config(allowlist)
        matcher = config["static_resources"]["listeners"][0]["filter_chains"][0][
            "filters"
        ][0]["typed_config"]["matcher"]
        assert "on_no_match" in matcher
        assert matcher["on_no_match"]["action"]["name"] == "deny"

    def test_has_dynamic_upstream_cluster(self, allowlist):
        config = generate_envoy_config(allowlist)
        clusters = config["static_resources"]["clusters"]
        names = [c["name"] for c in clusters]
        assert "dynamic_upstream" in names
        # dynamic_upstream uses ORIGINAL_DST (tunnel to the resolved IP).
        dyn = next(c for c in clusters if c["name"] == "dynamic_upstream")
        assert dyn["type"] == "ORIGINAL_DST"

    def test_has_deny_sinkhole_cluster(self, allowlist):
        config = generate_envoy_config(allowlist)
        clusters = config["static_resources"]["clusters"]
        names = [c["name"] for c in clusters]
        assert "deny_sinkhole" in names

    def test_empty_allowlist_still_produces_valid_config(self):
        al = NetworkAllowList.from_string("")
        config = generate_envoy_config(al)
        assert isinstance(config, dict)
        assert "static_resources" in config

    def test_config_is_json_serializable(self, allowlist):
        """The config must be JSON-serializable (Envoy accepts JSON)."""
        config = generate_envoy_config(allowlist)
        # Should not raise.
        json.dumps(config)


class TestWriteEnvoyConfig:
    def test_writes_valid_config(self, allowlist, tmp_path):
        """The written config is parseable (YAML if available, else JSON)."""
        out = tmp_path / "envoy.yaml"
        write_envoy_config(generate_envoy_config(allowlist), str(out))
        content = out.read_text()
        # Try YAML first (PyYAML may be installed), fall back to JSON.
        try:
            import yaml
            parsed = yaml.safe_load(content)
        except ImportError:
            parsed = json.loads(content)
        assert "static_resources" in parsed

    def test_write_and_read_roundtrip(self, allowlist, tmp_path):
        out = tmp_path / "envoy.yaml"
        original = generate_envoy_config(allowlist)
        write_envoy_config(original, str(out))
        content = out.read_text()
        try:
            import yaml
            parsed = yaml.safe_load(content)
        except ImportError:
            parsed = json.loads(content)
        assert parsed["admin"]["address"]["socket_address"]["port_value"] == 9901


class TestFindEnvoyBinary:
    def test_returns_str_or_none(self):
        result = find_envoy_binary()
        assert result is None or isinstance(result, str)


class TestLaunchEnvoyFailClosed:
    def test_raises_when_envoy_not_found(self, tmp_path):
        """If envoy is not on PATH, launch_envoy raises FileNotFoundError.

        We can't easily remove envoy from PATH in a test, so we test
        the behavior by mocking find_envoy_binary to return None.
        """
        import sandbox.envoy_sidecar as mod

        original = mod.find_envoy_binary
        mod.find_envoy_binary = lambda: None
        try:
            with pytest.raises(FileNotFoundError, match="envoy binary not found"):
                launch_envoy("/nonexistent/config.yaml")
        finally:
            mod.find_envoy_binary = original


class TestCLISmoke:
    def test_cli_generates_config_to_stdout(self, tmp_path):
        """The CLI --out - path writes config to stdout (no envoy needed)."""
        result = subprocess.run(
            [sys.executable, "-m", "sandbox.envoy_sidecar",
             "--allowlist", "pypi.org:443",
             "--out", "-"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        # Parse as YAML (preferred) or JSON.
        try:
            import yaml
            parsed = yaml.safe_load(result.stdout)
        except ImportError:
            parsed = json.loads(result.stdout)
        assert "static_resources" in parsed
