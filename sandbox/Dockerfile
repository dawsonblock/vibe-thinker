# vibe-thinker sandbox image — hardened Python execution environment
#
# This image is purpose-built for the DockerSandboxExecutor. It includes:
#   - Python 3.12 (slim base)
#   - iptables + ip6tables (for network allow-list enforcement)
#   - curl (for integration tests that verify network denial)
#   - A non-root user 'sandbox' (uid 1000) for candidate code execution
#   - An entrypoint script that applies firewall rules BEFORE candidate code
#
# The image is designed so that:
#   1. iptables is already installed (no apt-get at runtime — closes the
#      TOCTOU window where the container has open network before lockdown)
#   2. The entrypoint applies firewall rules, then drops privileges to
#      the 'sandbox' user, then exec's the candidate code
#   3. The candidate code runs as non-root with no NET_ADMIN — it cannot
#      modify the firewall rules that were applied before it started
#
# Build: docker build -t vibe-thinker-sandbox:latest -f sandbox/Dockerfile .
# Tag:   vibe-thinker-sandbox:latest

FROM python:3.12-slim

# Install iptables, ip6tables, and curl (for integration tests).
# --no-install-recommends keeps the image small.
# We clean apt lists to reduce image size.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        iptables \
        iproute2 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for candidate code execution.
# The entrypoint runs as root (needed for iptables), applies firewall
# rules, then drops to this user via `su-exec` or `runuser`.
RUN useradd --uid 1000 --create-home --shell /bin/false sandbox

# Copy the entrypoint script.
COPY sandbox/entrypoint.sh /usr/local/bin/vt-entrypoint.sh
RUN chmod +x /usr/local/bin/vt-entrypoint.sh

# Default entrypoint — the executor overrides the command but uses
# this entrypoint to ensure firewall rules are applied first.
ENTRYPOINT ["/usr/local/bin/vt-entrypoint.sh"]

# Default command (can be overridden by docker run args).
CMD ["python3", "--version"]
