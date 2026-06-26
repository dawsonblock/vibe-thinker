#!/bin/bash
# vibe-thinker sandbox entrypoint
#
# This script runs as root (the container starts as root via Docker).
# It applies firewall rules from an environment variable, then drops
# privileges to the 'sandbox' user and execs the candidate command.
#
# The firewall rules are passed via the VT_IPTABLES_RULES environment
# variable as a base64-encoded, newline-separated list of iptables
# commands. The executor sets this before starting the container.
#
# If VT_IPTABLES_RULES is empty/unset, no firewall rules are applied
# (the container uses --network=none in that case, so there's no network
# to filter).
#
# If VT_IPTABLES_RULES is set to "__DENY_ALL__", a default deny-all
# policy is applied (used when the allow-list is empty but --network=default
# is active for some reason).
#
# Security properties:
#   1. Firewall rules are applied BEFORE candidate code runs (no TOCTOU)
#   2. Candidate code runs as uid 1000 (sandbox user), NOT root
#   3. Candidate code has no NET_ADMIN capability (dropped by Docker)
#   4. IPv6 is disabled via ip6tables DROP policy (prevents IPv6 bypass)
#   5. DNS is restricted to the resolver in /etc/resolv.conf (if specified)
#
set -euo pipefail

# --- Phase 1: Apply firewall rules (as root) ---

IPTABLES_RULES_B64="${VT_IPTABLES_RULES:-}"
DNS_RESOLVER="${VT_DNS_RESOLVER:-}"

if [ -n "$IPTABLES_RULES_B64" ]; then
    if [ "$IPTABLES_RULES_B64" = "__DENY_ALL__" ]; then
        # Deny all egress (empty allow-list with --network=default).
        iptables -P OUTPUT DROP
        iptables -A OUTPUT -o lo -j ACCEPT
        iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    else
        # Decode and apply the allow-list rules.
        RULES=$(echo "$IPTABLES_RULES_B64" | base64 -d)
        while IFS= read -r rule; do
            [ -z "$rule" ] && continue
            # Apply the rule. Errors here are fatal — fail-closed.
            eval "$rule" || {
                echo "[vt-entrypoint] FATAL: iptables rule failed: $rule" >&2
                exit 1
            }
        done <<< "$RULES"
    fi

    # --- IPv6: deny all by default ---
    # The allow-list rules only cover IPv4 (iptables). To prevent IPv6
    # bypass, we set the ip6tables OUTPUT policy to DROP and allow only
    # loopback + established. This is the safest default — if IPv6 is
    # needed, it should be explicitly allow-listed.
    ip6tables -P OUTPUT DROP 2>/dev/null || true
    ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
    ip6tables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true

    # --- DNS restriction ---
    # If a specific DNS resolver is specified, restrict DNS (port 53) to
    # that resolver only. Otherwise, DNS is already handled by the
    # iptables rules (the allow-list includes DNS allow rules).
    if [ -n "$DNS_RESOLVER" ]; then
        # Remove the broad DNS allow rules (if present) and add
        # resolver-specific ones.
        iptables -D OUTPUT -p udp --dport 53 -j ACCEPT 2>/dev/null || true
        iptables -D OUTPUT -p tcp --dport 53 -j ACCEPT 2>/dev/null || true
        iptables -A OUTPUT -d "$DNS_RESOLVER" -p udp --dport 53 -j ACCEPT
        iptables -A OUTPUT -d "$DNS_RESOLVER" -p tcp --dport 53 -j ACCEPT
    fi

    # Log the final firewall state (for audit purposes).
    echo "[vt-entrypoint] Firewall rules applied:" >&2
    iptables -L OUTPUT -n --line-numbers >&2 || true
    if [ -n "$DNS_RESOLVER" ]; then
        echo "[vt-entrypoint] DNS restricted to: $DNS_RESOLVER" >&2
    fi
    echo "[vt-entrypoint] IPv6 OUTPUT policy: DROP" >&2
fi

# --- Phase 2: Drop privileges and exec candidate code ---

# If VT_NO_DROP is set, keep running as root (for debugging only).
# This is NEVER set in production — the executor always drops privileges.
if [ "${VT_NO_DROP:-}" = "1" ]; then
    echo "[vt-entrypoint] WARNING: VT_NO_DROP=1 — running as root (DEBUG ONLY)" >&2
    exec "$@"
fi

# Drop to the sandbox user (uid 1000) and exec the candidate command.
# We use 'runuser' (available on Debian-based images) to switch user.
# The candidate code inherits no capabilities (NET_ADMIN is dropped by
# Docker's --cap-drop=ALL), so it cannot modify the firewall.
exec runuser -u sandbox -- "$@"
