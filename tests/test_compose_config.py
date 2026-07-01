"""Static checks on docker-compose.yml — no Docker daemon required.

Validates the security-hardened shape of the sni-proxy service and the
proxy/Envoy configuration documented in the compose file.
"""

from pathlib import Path

import yaml


def test_sni_proxy_service_is_hardened():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    svc = compose["services"]["sni-proxy"]
    assert svc["read_only"] is True
    assert "ALL" in svc["cap_drop"]
    assert "no-new-privileges:true" in svc["security_opt"]
    assert svc["user"] == "1000:1000"
    assert svc["pids_limit"] == 64
    assert svc["mem_limit"] == "128m"
    assert "/tmp:rw,size=10m" in svc["tmpfs"]


def test_compose_uses_python_sni_proxy_not_envoy_for_http_proxy():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    env = compose["x-orchestrator-env"]
    assert "sni-proxy:8888" in env["RFSN_PROXY_EGRESS"]
    assert env["RFSN_ENVOY_SIDECAR"] in ("", "${RFSN_ENVOY_SIDECAR:-}")


def test_sni_proxy_not_public_unless_intentional():
    svc = yaml.safe_load(Path("docker-compose.yml").read_text())["services"]["sni-proxy"]
    assert "ports" not in svc
    assert "8888" in svc.get("expose", [])


def test_compose_configures_sandbox_docker_network():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    env = compose["x-orchestrator-env"]
    assert "vibe-thinker_default" in env["RFSN_DOCKER_NETWORK"]
    networks = compose.get("networks", {})
    assert networks.get("default", {}).get("name") == "vibe-thinker_default"
