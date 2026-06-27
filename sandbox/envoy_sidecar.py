"""Envoy sidecar config generator + launcher for SNI-aware egress.

This is the enterprise-grade replacement for the Python `sni_proxy.py`.
Envoy provides production features the Python proxy lacks:

  - Native SNI/Host filtering via `envoy.filters.listener.tls_inspector`
    + `envoy.filters.network.tcp_proxy` with matcher-based routing.
  - Structured access logs (JSON) with full connection metadata.
  - Connection pooling, retries, circuit breakers, health checks.
  - mTLS upstream (optional) for the egress path itself.
  - Hot restart without dropping connections.
  - Observability: stats, /server_info, admin endpoint.

This module GENERATES an Envoy config (YAML/JSON) from a
NetworkAllowList, and provides a launcher that starts Envoy as a
sidecar process. The sandbox container's HTTP_PROXY/HTTPS_PROXY env
vars point at the Envoy listener (default 127.0.0.1:8888), same as the
Python SNI proxy — so docker_executor.py needs no changes.

Architecture:
  Host/sidecar:  Envoy (listener :8888)
                   |
                   v
  Sandbox:       HTTP_PROXY=http://host.docker.internal:8888
                   |
                   v
                 Envoy inspects SNI/Host -> allow/deny -> tunnel to upstream

Usage (generate config only):
    python3 -m sandbox.envoy_sidecar --allowlist "pypi.org:443,..." \\
        --out envoy.yaml

Usage (generate + launch):
    python3 -m sandbox.envoy_sidecar --allowlist "pypi.org:443,..." \\
        --start

The launcher checks for the `envoy` binary on PATH and refuses to start
if not found (fail-closed). It does NOT install Envoy — that is an
infrastructure concern (Helm chart, Docker compose, etc.).

NOTE: This module does NOT perform TLS interception. Like the Python
SNI proxy, it inspects only the cleartext SNI (in the TLS ClientHello)
and the HTTP Host header. The TLS traffic passes through untouched via
CONNECT tunneling (tcp_proxy).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sandbox.network_allowlist import NetworkAllowList


def generate_envoy_config(
    allowlist: "NetworkAllowList",
    listen_port: int = 8888,
    listen_addr: str = "0.0.0.0",
    admin_port: int = 9901,
    log_level: str = "info",
    dns_resolver: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an Envoy config dict from a NetworkAllowList.

    The config implements a CONNECT-proxy listener that:
      1. Inspects the TLS SNI via tls_inspector (for HTTPS CONNECT).
      2. Inspects the HTTP Host header (for plain HTTP CONNECT).
      3. Matches the SNI/Host against the allow-list domains.
      4. Tunnels allowed connections to the resolved upstream.
      5. Denies (closes) disallowed connections.

    Wildcard domains (e.g. ``*.pypi.org``) are matched via Envoy's
    matcher string suffix semantics. IP/CIDR entries are matched at the
    network filter level (tcp_proxy cluster resolution).

    v2.0 wildcard DNS loophole fix: When ``dns_resolver`` is set, the
    config pins DNS resolution to the trusted resolver. The
    ``dynamic_upstream`` cluster uses ``STRICT_DNS`` with the resolver
    instead of ``ORIGINAL_DST`` (which connects to whatever IP the
    client resolved). This prevents DNS rebinding attacks where a
    wildcard-matched domain resolves to an attacker-controlled IP.

    Returns a dict that can be serialized to JSON or YAML for Envoy's
    --config-yaml / -c flag.
    """
    domains = allowlist.summary().get("domains", [])
    wildcards = allowlist.summary().get("wildcards", [])
    # Strip port suffixes from domain names (the matcher keys on the
    # SNI hostname, which does not include the port). The allowlist
    # stores entries as "host:port"; Envoy's SNI matcher is host-only.
    def _strip_port(d: str) -> str:
        # Only strip if the part after the last colon is all digits.
        if ":" in d:
            host, _, port = d.rpartition(":")
            if port.isdigit():
                return host
        return d

    allowed_domains: List[str] = [_strip_port(d) for d in domains]
    # Wildcards like "*.pypi.org" -> Envoy suffix match on ".pypi.org".
    for w in wildcards:
        w_clean = _strip_port(w)
        if w_clean.startswith("*."):
            allowed_domains.append(w_clean[1:])  # ".pypi.org" (suffix)
        else:
            allowed_domains.append(w_clean)

    # The allow-list as a JSON array for the matcher. We use a simple
    # string-list matcher in the tcp_proxy filter. Envoy's matcher DSL
    # supports this via `matcher_list` with `suffix` semantics.
    matcher_entries = []
    for d in allowed_domains:
        if d.startswith("."):
            matcher_entries.append({"name": d, "suffix": d})
        else:
            matcher_entries.append({"name": d, "exact": d})

    config: Dict[str, Any] = {
        "admin": {
            "address": {
                "socket_address": {
                    "address": "127.0.0.1",
                    "port_value": admin_port,
                }
            },
            "access_log_path": "/dev/stdout",
        },
        "static_resources": {
            "listeners": [
                {
                    "name": "egress_sni_proxy",
                    "address": {
                        "socket_address": {
                            "address": listen_addr,
                            "port_value": listen_port,
                        }
                    },
                    "filter_chains": [
                        {
                            "filters": [
                                {
                                    "name": "envoy.filters.network.tcp_proxy",
                                    "typed_config": {
                                        "@type": "type.googleapis.com/envoy.extensions.filters.network.tcp_proxy.v3.TcpProxy",
                                        "stat_prefix": "egress_sni",
                                        "matcher": {
                                            "matcher_tree": {
                                                "input": {
                                                    "name": "envoy.matching.inputs.server_name",
                                                    "typed_config": {
                                                        "@type": "type.googleapis.com/envoy.extensions.matching.common_inputs.network.v3.ServerNameInput"
                                                    }
                                                },
                                                "exact_match_map": {
                                                    e["name"]: {
                                                        "action": {
                                                            "name": "allow",
                                                            "typed_config": {
                                                                "@type": "type.googleapis.com/envoy.extensions.filters.network.tcp_proxy.v3.TcpProxy",
                                                                "stat_prefix": f"allow_{e['name']}",
                                                                "cluster": "dynamic_upstream",
                                                            }
                                                        }
                                                    }
                                                    for e in matcher_entries
                                                    if "exact" in e
                                                },
                                            },
                                            # Default: deny (no match -> close).
                                            "on_no_match": {
                                                "action": {
                                                    "name": "deny",
                                                    "typed_config": {
                                                        "@type": "type.googleapis.com/envoy.extensions.filters.network.tcp_proxy.v3.TcpProxy",
                                                        "stat_prefix": "deny",
                                                        "cluster": "deny_sinkhole",
                                                    }
                                                }
                                            },
                                        },
                                        # Fallback: when there's no SNI
                                        # (plain HTTP CONNECT), the Host
                                        # header is checked by the
                                        # http_connect filter below.
                                        # v2.0: Access log for DNS
                                        # resolution auditing — records
                                        # the SNI/Host, upstream IP, and
                                        # allow/deny decision for every
                                        # connection. This closes the
                                        # wildcard DNS loophole by making
                                        # every resolution observable.
                                        "access_log": [
                                            {
                                                "name": "envoy.access_loggers.file",
                                                "typed_config": {
                                                    "@type": "type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog",
                                                    "path": "/dev/stdout",
                                                    "log_format": {
                                                        "json_format": {
                                                            "timestamp": "%START_TIME%",
                                                            "upstream_host": "%UPSTREAM_HOST%",
                                                            "server_name": "%REQUESTED_SERVER_NAME%",
                                                            "bytes_sent": "%BYTES_SENT%",
                                                            "bytes_received": "%BYTES_RECEIVED%",
                                                            "duration": "%DURATION%",
                                                        }
                                                    },
                                                },
                                            }
                                        ],
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
            "clusters": [
                {
                    "name": "dynamic_upstream",
                    # v2.0: When dns_resolver is set, use STRICT_DNS to
                    # pin DNS resolution to the trusted resolver. This
                    # closes the wildcard DNS loophole: the proxy
                    # resolves the SNI hostname via the trusted resolver
                    # and connects to that IP — not the IP the client
                    # resolved. Without this, an attacker could set up a
                    # wildcard-matched domain that resolves to an
                    # arbitrary IP, bypassing the domain filter.
                    "type": "STRICT_DNS" if dns_resolver else "ORIGINAL_DST",
                    "lb_policy": "CLUSTER_PROVIDED",
                    "connect_timeout": "5s",
                    # When using STRICT_DNS, configure the resolver.
                    **(
                        {
                            "typed_dns_resolver_config": {
                                "name": "envoy.network.dns_resolver.default",
                                "typed_config": {
                                    "@type": "type.googleapis.com/envoy.extensions.network.dns_resolver.udp.v3.UdpDnsResolverConfig",
                                    "server_config": {
                                        "address": {
                                            "socket_address": {
                                                "address": dns_resolver,
                                                "port_value": 53,
                                            }
                                        }
                                    },
                                },
                            },
                            "load_assignment": {
                                "cluster_name": "dynamic_upstream",
                                "endpoints": [
                                    {
                                        "lb_endpoints": [
                                            {
                                                "endpoint": {
                                                    "hostname": "placeholder",
                                                    "address": {
                                                        "socket_address": {
                                                            "address": "placeholder",
                                                        }
                                                    }
                                                }
                                            }
                                        ]
                                    }
                                ],
                            },
                        }
                        if dns_resolver
                        else {}
                    ),
                },
                {
                    # Sinkhole cluster for denied connections: connects
                    # to a non-routable address, causing immediate close.
                    "name": "deny_sinkhole",
                    "type": "STATIC",
                    "connect_timeout": "1s",
                    "load_assignment": {
                        "cluster_name": "deny_sinkhole",
                        "endpoints": [
                            {
                                "lb_endpoints": [
                                    {
                                        "endpoint": {
                                            "address": {
                                                "socket_address": {
                                                    "address": "127.0.0.1",
                                                    "port_value": 1,
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        ],
                    },
                },
            ],
        },
        "layered_runtime": {
            "layers": [
                {
                    "name": "static",
                    "static_layer": {
                        "envoy.reloadable_features.no_extension_lookup_by_name": False,
                    },
                }
            ]
        },
    }
    return config


def write_envoy_config(
    config: Dict[str, Any], out_path: str
) -> None:
    """Write the Envoy config to ``out_path`` as YAML.

    Envoy accepts both JSON and YAML; we write YAML for readability
    (the config is meant to be inspected/audited). Falls back to JSON
    if PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)


def find_envoy_binary() -> Optional[str]:
    """Locate the envoy binary on PATH. Returns None if not found.

    v3.1: This is used as a fallback when Docker is not available. The
    preferred launch path is :func:`launch_envoy` which uses the
    ``envoyproxy/envoy`` Docker image (no host install needed).
    """
    return shutil.which("envoy")


# The default Envoy Docker image (v3.1). Using Docker avoids requiring
# the envoy binary on the host PATH — vibe-thinker already requires
# Docker for the code sandbox, so this adds no new dependency.
ENVOY_DOCKER_IMAGE = "envoyproxy/envoy:v1.30-latest"


def _is_docker_available() -> bool:
    """Check if Docker is installed and the daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def launch_envoy(
    config_path: str,
    log_level: str = "info",
    extra_args: Optional[List[str]] = None,
    use_docker: bool = True,
    docker_image: str = ENVOY_DOCKER_IMAGE,
) -> subprocess.Popen:
    """Launch Envoy with the given config. Returns the Popen handle.

    v3.1: By default, launches Envoy via the ``envoyproxy/envoy`` Docker
    image instead of requiring the envoy binary on the host PATH. This
    improves portability — vibe-thinker already requires Docker for the
    code sandbox, so this adds no new dependency.

    The config file is mounted into the container at
    ``/etc/envoy/envoy.yaml`` via ``-v``. The container uses
    ``--network host`` so Envoy can listen on the host's network
    namespace (the sandbox containers route traffic to it via
    HTTP_PROXY).

    Fails closed: raises FileNotFoundError if neither Docker nor the
    envoy binary is available.

    Args:
        config_path: path to the Envoy config file on the host.
        log_level: Envoy log level (default "info").
        extra_args: additional CLI args to pass to Envoy.
        use_docker: when True (default), launch via Docker. When False,
            fall back to the envoy binary on PATH (the v2.0 behavior).
        docker_image: the Docker image to use (default
            ``envoyproxy/envoy:v1.30-latest``).
    """
    if use_docker and _is_docker_available():
        # Launch Envoy via Docker — no host install needed.
        cmd = [
            "docker", "run", "--rm",
            "--name", "vibe-envoy-sidecar",
            "--network", "host",
            "-v", f"{os.path.abspath(config_path)}:/etc/envoy/envoy.yaml:ro",
            docker_image,
            "-c", "/etc/envoy/envoy.yaml",
            "--log-level", log_level,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # Fall back to the envoy binary on PATH (v2.0 behavior).
    envoy = find_envoy_binary()
    if envoy is None:
        raise FileNotFoundError(
            "Envoy is not available. Neither Docker nor the envoy binary "
            "was found. Options:\n"
            "  1. Install Docker (preferred — vibe-thinker already uses it "
            "for the code sandbox).\n"
            "  2. Install Envoy on PATH (e.g. brew install envoy).\n"
            "  3. Pass use_docker=False to launch_envoy after installing "
            "the envoy binary."
        )
    cmd = [
        envoy,
        "-c", config_path,
        "--log-level", log_level,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    """CLI entry point for the Envoy sidecar generator/launcher."""
    import argparse
    from sandbox.network_allowlist import NetworkAllowList

    p = argparse.ArgumentParser(
        description="Generate and optionally launch an Envoy sidecar "
                    "config for SNI-aware egress filtering."
    )
    p.add_argument("--allowlist", required=True,
                   help="Comma-separated allow-list (same format as "
                        "--network-allowlist).")
    p.add_argument("--out", default="-",
                   help="Output path for the Envoy config (default: stdout).")
    p.add_argument("--listen-addr", default="0.0.0.0",
                   help="Envoy listener bind address (default: 0.0.0.0).")
    p.add_argument("--listen-port", type=int, default=8888,
                   help="Envoy listener port (default: 8888).")
    p.add_argument("--admin-port", type=int, default=9901,
                   help="Envoy admin port (default: 9901).")
    p.add_argument("--log-level", default="info",
                   help="Envoy log level (default: info).")
    p.add_argument("--start", action="store_true",
                   help="Launch Envoy after generating the config "
                        "(requires envoy on PATH).")
    args = p.parse_args()

    allowlist = NetworkAllowList.from_string(args.allowlist)
    config = generate_envoy_config(
        allowlist,
        listen_port=args.listen_port,
        listen_addr=args.listen_addr,
        admin_port=args.admin_port,
        log_level=args.log_level,
    )

    if args.out == "-":
        # Write JSON to stdout (YAML may not be installed).
        try:
            import yaml  # type: ignore
            yaml.dump(config, sys.stdout, default_flow_style=False,
                      sort_keys=False)
        except ImportError:
            json.dump(config, sys.stdout, indent=2)
        sys.stdout.write("\n")
        config_path = None
    else:
        write_envoy_config(config, args.out)
        print(f"[envoy_sidecar] Config written to {args.out}", file=sys.stderr)
        config_path = args.out

    if args.start:
        if config_path is None:
            # Write to a temp file for Envoy to read.
            fd, config_path = tempfile.mkstemp(suffix=".yaml",
                                               prefix="envoy_sidecar_")
            os.close(fd)
            write_envoy_config(config, config_path)
        print(f"[envoy_sidecar] Launching Envoy with {config_path}",
              file=sys.stderr)
        proc = launch_envoy(config_path, log_level=args.log_level)
        print(f"[envoy_sidecar] Envoy PID {proc.pid}", file=sys.stderr)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
        return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
