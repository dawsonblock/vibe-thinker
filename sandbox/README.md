# Sandbox Execution Layer

The `sandbox/` package provides isolated code execution for CodeVerifier.
It implements defense-in-depth: the sbx microVM is the outer layer, and
a Docker container is the inner layer.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Docker Sandbox (sbx microVM)               │  ← outer layer
│  ┌───────────────────────────────────────┐  │
│  │  vibe-thinker orchestrator            │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  CodeVerifier                   │  │  │
│  │  │  ┌───────────────────────────┐  │  │  │
│  │  │  │  DockerSandboxExecutor    │  │  │  │  ← inner layer
│  │  │  │  docker run --network=none │  │  │  │
│  │  │  │  --memory=128m --read-only │  │  │  │
│  │  │  │  --security-opt=no-new-... │  │  │  │
│  │  │  └───────────────────────────┘  │  │
│  │  └─────────────────────────────────┘  │
│  └───────────────────────────────────────┘
└─────────────────────────────────────────────┘
```

## Executors

### DockerSandboxExecutor (inner layer)

Runs Python in a Docker container with hardening:
- `--network=none` (no network access by default)
- `--memory=128m` (memory limit)
- `--read-only` (read-only root filesystem)
- `--security-opt=no-new-privileges` (no privilege escalation)
- `--cap-drop=ALL` (drop all Linux capabilities)
- `--pids-limit=64` (process count limit)
- `--tmpfs /tmp` (writable temp dir for execution)
- `--rm` (auto-remove container after exit)

This is the default executor for CodeVerifier. It is lightweight
(containers start in <1s) and provides strong isolation for code
verification. Note: enforced egress (NetworkMode.ENFORCED_GATEWAY) is
experimental and not production-safe; DISABLED and BEST_EFFORT_PROXY
modes are convenience routing, not a security boundary.

### DockerSbxExecutor (outer layer)

Runs code in an sbx microVM with full VM isolation:
- Separate kernel per sandbox
- No shared memory/processes with host
- Private Docker Engine inside sandbox
- No path to host Docker daemon
- Network traffic mediated by host proxy
- Credentials injected by proxy, not copied into VM

This is heavier weight (microVM startup) but provides the strongest
isolation boundary. Use it when running untrusted agent code that needs
full VM isolation, or when the agent needs its own Docker daemon.

### LocalSubprocessExecutor (development only)

Runs Python in a local subprocess with timeout only. **NOT safe for
untrusted code.** No filesystem, network, or memory isolation.

Only used when:
- Docker is not available AND `allow_unsafe=True` is explicitly passed
- Running trusted test code in CI

## Selection Policy

`select_executor()` picks the best available backend:

1. `DockerSbxExecutor` (if `prefer_sbx=True` and sbx available)
2. `DockerSandboxExecutor` (if Docker available)
3. `DockerSbxExecutor` (if sbx available)
4. `LocalSubprocessExecutor` (ONLY if `allow_unsafe=True`)
5. `None` → refuse verification

The verifier **refuses to run** if no sandbox is available and
`allow_unsafe=False`. It does not fall back to host execution for
untrusted code.

## Sandbox network status

The default safe mode is `DISABLED`, which runs candidate code with
Docker `--network none`.

`BEST_EFFORT_PROXY` is not a security boundary. It only affects clients
that respect proxy environment variables. Code may bypass it with raw
sockets, direct IP connections, custom DNS, or clients that ignore
proxy variables.

`ENFORCED_GATEWAY` starts a gateway container running the SNI egress
proxy. Docker network isolation is tested (`--network none` and
`--internal` networks block connections to the internet, cloud metadata,
and host LAN), and the allowlisted gateway/proxy egress path is
validated: allowlisted domains are reachable through the proxy,
non-allowlisted domains are blocked (403), and raw socket egress
(bypassing the proxy) is blocked by the `--internal` network.

Do not run hostile code with network access enabled.

## Running vibe-thinker in an sbx microVM

```bash
# Install sbx
brew install docker/tap/sbx
sbx login

# Set locked-down network policy (recommended for autonomous work)
sbx policy set-default deny-all

# Allow only the model API endpoint
sbx policy allow network api.anthropic.com  # or your model provider

# Run vibe-thinker in clone mode (agent writes only inside sandbox clone)
cd ~/vibe-thinker
sbx run --clone --name vibe-thinker-sandbox shell

# Inside the sandbox, the agent has its own Docker daemon
# CodeVerifier's DockerSandboxExecutor will use it for code verification
```

## Hard Policy Defaults

- **Clone mode required** — agent writes only inside sandbox clone
- **deny-all network by default** — network is opt-in, never default
- **No global secrets for code verification** — use sandbox-scoped secrets
- **Timeout required** — no unlimited execution
- **Fetch sandbox remote and diff before accepting work**
- **Never trust direct mode for autonomous edits**
