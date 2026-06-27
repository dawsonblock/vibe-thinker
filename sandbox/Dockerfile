# vibe-thinker sandbox image — hardened Python execution environment
#
# This image is purpose-built for the DockerSandboxExecutor. It includes:
#   - Python 3.12 (slim base)
#   - iptables + ip6tables (legacy in-container firewall; DEPRECATED in
#     favor of host-level SNI-proxy / Envoy egress — kept for opt-in
#     fallback via `docker run --user root`)
#   - curl (for integration tests that verify network denial)
#   - A non-root user 'sandbox' (uid 1000) — the DEFAULT runtime user
#
# v1.0 privilege model (Phase 1.3):
#   - The image defaults to USER sandbox (uid 1000). The container
#     lifecycle no longer requires root — the production path
#     (DockerSandboxExecutor) runs python3 directly as uid 1000 with
#     --cap-drop ALL and --security-opt no-new-privileges.
#   - The legacy iptables entrypoint is root-aware: it only applies
#     firewall rules when invoked AS root (opt-in via
#     `docker run --user root`), otherwise it exec's the command
#     directly as the current (non-root) user.
#   - Network egress filtering is handled at the HOST level by the
#     SNI proxy / Envoy sidecar (Phase 1.2), not in-container iptables.
#
# Build: docker build -t vibe-thinker-sandbox:latest -f sandbox/Dockerfile .
# Tag:   vibe-thinker-sandbox:latest

FROM python:3.12-slim

# Install iptables, ip6tables, and curl (for integration tests).
# --no-install-recommends keeps the image small.
# We clean apt lists to reduce image size.
# NOTE: iptables is only used by the DEPRECATED legacy entrypoint path
# (opt-in via `docker run --user root`). The production proxy path does
# not need it, but it is retained for defense-in-depth fallback.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        iptables \
        iproute2 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for candidate code execution.
# v1.0: this user is the DEFAULT runtime user (see USER below). The
# container no longer starts as root.
RUN useradd --uid 1000 --create-home --shell /bin/false sandbox

# Copy the entrypoint script.
COPY sandbox/entrypoint.sh /usr/local/bin/vt-entrypoint.sh
RUN chmod +x /usr/local/bin/vt-entrypoint.sh

# Default entrypoint — the executor overrides the command but uses
# this entrypoint to ensure firewall rules are applied first (when run
# as root; as non-root it just exec's the command).
ENTRYPOINT ["/usr/local/bin/vt-entrypoint.sh"]

# v1.0 (Phase 1.3): run as the non-root sandbox user by default. The
# production executor also passes --user 1000:1000 explicitly. The
# legacy iptables entrypoint path requires `docker run --user root` to
# opt back into root (deprecated).
USER sandbox

# Default command (can be overridden by docker run args).
CMD ["python3", "--version"]
