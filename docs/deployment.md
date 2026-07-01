# vibe-thinker Docker deployment

This repo ships a unified Docker Compose stack that bundles the sidecars the
orchestrator normally needs you to run by hand: **Redis** (federation
heartbeat/reaping + web UI Pub/Sub), **Python SNI proxy** (`sni-proxy`,
HTTP/HTTPS CONNECT egress proxy), and **AgentDB** (vector store). You pick
an LLM topology with a single `--profile` flag.

The Envoy sidecar (`--envoy-sidecar` / `sandbox/envoy_sidecar.py`) is for
standalone transparent-routing experiments on the host and is **not** used as
the HTTP proxy in the compose stack.

## Files

| File | Purpose |
| --- | --- |
| `docker-compose.yml` | Bundles sidecars + orchestrator with 3 profiles. |
| `Dockerfile` | Multi-stage Python 3.12 image for the orchestrator. |
| `docker/envoy/envoy.yaml` | Minimal Envoy egress-proxy config. |
| `docker/agentdb.Dockerfile` | Placeholder AgentDB image (see below). |
| `.dockerignore` | Keeps the image small. |

## Prerequisites

- Docker + Docker Compose v2 (`docker compose`).
- For the `gpu` profile: NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
- A `.gguf` model file for the `gpu` / `ruvllm` profiles (mounted via volume,
  never baked into the image).

## Quick start — CPU profile (external/host LLM)

The CPU profile runs the orchestrator only and points it at an LLM server you
already have running (e.g. a `llama-server` on the host or a remote endpoint).

```bash
# Point at a host-side llama-server (specialist on :8080, generalist on :8081).
# host.docker.internal resolves to the Docker host from inside the container.
VIBE_THINKER_URL=http://host.docker.internal:8080 \
GENERALIST_URL=http://host.docker.internal:8081 \
docker compose --profile cpu up
```

The sidecars (redis, envoy, agentdb) always start regardless of profile.

## Quick start — GPU profile (bundled llama.cpp server)

The GPU profile adds a `llama-server` container with a CUDA GPU reservation.
Mount your model directory and set the model path:

```bash
# Put your model at ./models/model.gguf (or set LLAMA_MODEL_DIR).
LLAMA_MODEL_DIR=$HOME/models \
LLAMA_MODEL_PATH=/models/my-model.gguf \
LLAMA_N_GPU_LAYERS=99 \
docker compose --profile gpu up
```

The orchestrator is wired to point `VIBE_THINKER_URL` and `GENERALIST_URL` at
the bundled `llama-server` service automatically.

> **Note:** The `ghcr.io/llama.cpp/llama-server:latest` image is public but
> you should pin a specific digest/tag for production. You MUST supply your
> own `.gguf` model — no model is included.

## Quick start — RuvLLM profile

RuvLLM is a Rust inference engine with TurboQuant KV cache compression that
exposes the same OpenAI-compatible `/completion` API as `llama-server`.

```bash
# Build the RuvLLM image first (no published image exists):
#   cd ruvllm_py && cargo build --release --features candle
# Then wrap it in an image and set RUVLLM_IMAGE:
RUVLLM_IMAGE=ruvllm:latest \
RUVLLM_MODEL_DIR=$HOME/models \
RUVLLM_MODEL_PATH=/models/my-model.gguf \
docker compose --profile ruvllm up
```

The orchestrator's `RUVLLM_URL` is wired to `http://ruvllm:8080`, which
overrides `--vibe` (the CLI gives `--ruvllm-url` precedence over `--vibe`).

> **Note:** There is no published RuvLLM Docker image. The `build:` context
> in `docker-compose.yml` points at `ruvllm_py/` as a placeholder. You must
> build the Rust crate and supply a real image via `RUVLLM_IMAGE` before this
> profile will actually serve inference.

## Environment variables

All orchestrator CLI flags are read from environment variables (precedence:
explicit CLI args > env vars > defaults — see `rfsn_cli.py:build_argparser`).
The most important ones:

| Env var | CLI flag | Default | Purpose |
| --- | --- | --- | --- |
| `VIBE_THINKER_URL` | `--vibe` | `http://127.0.0.1:8080` | Specialist LLM endpoint |
| `GENERALIST_URL` | `--generalist` | `http://127.0.0.1:8081` | Generalist LLM endpoint |
| `CODE_SPECIALIST_URL` | `--code-specialist` | (empty) | Dedicated code-specialist endpoint |
| `RUVLLM_URL` | `--ruvllm-url` | (empty) | RuvLLM endpoint (overrides `--vibe`) |
| `AGENTDB_URL` | `--agentdb-url` | `http://agentdb:8088` | AgentDB vector store endpoint |
| `VIBE_THINKER_AGENTDB_ONLY` | `--agentdb-only` | `false` | AgentDB-only mode (no local fallback) |
| `RFSN_ENVOY_SIDECAR` | `--envoy-sidecar` | (empty) | Launch Envoy as a child process (host mode, not used in compose) |
| `RFSN_PROXY_EGRESS` | `--proxy-egress` | `sni-proxy:8888` | Sandbox egress proxy address (Python SNI proxy in compose) |
| `RFSN_NETWORK_ALLOWLIST` | `--network-allowlist` | `pypi.org:443,files.pythonhosted.org:443` | Allowed sandbox egress domains |
| `FEDERATION_URL` | `--federation-url` | (empty) | Federation coordinator URL (multi-node) |
| `RFSN_MAX_CONCURRENT` | `--max-concurrent` | `2` | Max concurrent jobs |
| `RFSN_USE_CLR` | `--clr` | `true` | Enable Claim-Level Reliability |
| `RFSN_CLR_K` | `--clr-k` | `8` | CLR k |
| `RFSN_USE_EMBEDDING_ROUTER` | `--embedding-router` | `true` | Embedding-based semantic routing |
| `RFSN_USE_TRAJECTORY_STORE` | `--trajectory-store` | `true` | Verified-trajectory memory |
| `TRAJECTORY_STORE_PATH` | `--trajectory-store-path` | `/data/verified_trajectories.json` | Trajectory store file |
| `RFSN_AUDIT_LOG` | `--audit-log` | `/data/rfsn_jobs_bitemporal.jsonl` | Bi-temporal audit log path |
| `VIBE_THINKER_LOCAL_MODEL` | `--local-specialist-model` | (empty) | In-process .gguf specialist path |
| `RFSN_SANDBOX_IMAGE` | `--sandbox-image` | `vibe-thinker-sandbox:latest` | Docker image for code sandbox |

Override any of these on the `docker compose` command line with `-e`:

```bash
docker compose --profile cpu up -e VIBE_THINKER_URL=http://10.0.0.5:8080
```

## Ports

| Port | Service | Purpose |
| --- | --- | --- |
| `6379` | redis | Redis (federation state + Pub/Sub) |
| `8080` | llama-server / ruvllm | Specialist LLM `/completion` endpoint |
| `8081` | llama-server | Generalist LLM `/completion` endpoint |
| `8088` | agentdb | AgentDB `POST /v1/vector/search` |
| `8888` | sni-proxy | Python SNI-aware egress proxy listener (internal only) |
| `9901` | envoy (host mode only) | Envoy admin interface (`--envoy-sidecar`, not used in compose) |

## Healthchecks

Every service has a healthcheck with `start_period` and `retries`:

| Service | Healthcheck | start_period | retries |
| --- | --- | --- | --- |
| `redis` | `redis-cli ping` | 10s | 5 |
| `sni-proxy` | TCP connect to `127.0.0.1:8888` | 15s | 5 |
| `agentdb` | `GET /health` (:8088) | 20s | 5 |
| `llama-server` | `GET /health` (:8080) | 60s | 6 |
| `ruvllm` | `GET /health` (:8080) | 60s | 6 |

The orchestrator services use `depends_on: condition: service_healthy` for
redis/envoy/llama-server/ruvllm, and `service_started` for agentdb (since the
placeholder starts immediately). The orchestrator itself fail-closes when a
sidecar is unreachable — it does not crash, it falls back to local behavior.

## Switching profiles

Profiles are mutually exclusive at the orchestrator level but the sidecars are
shared. To switch, stop the stack and bring it up with a different profile:

```bash
docker compose down
docker compose --profile gpu up
```

You can run sidecars alone (no orchestrator) by omitting the profile — useful
if you run the orchestrator on the host but want the sidecars containerized:

```bash
docker compose up redis sni-proxy agentdb
```

## AgentDB — important caveat

There is **no published AgentDB Docker image**. The `docker/agentdb.Dockerfile`
is a **placeholder** that starts a trivial HTTP server returning empty search
results so the compose stack and healthchecks pass. To get real vector search:

1. Build a real AgentDB image from [ruvnet/ruflo](https://github.com/ruvnet/ruflo)
   and set `image:` in `docker-compose.yml` (replacing the `build:` block), OR
2. Point `AGENTDB_URL` at a host-side AgentDB instance and remove the `agentdb`
   service from the compose file.

Until then, the orchestrator fail-closes: `AgentDBVectorStore` returns `[]`
when the sidecar is down and reads fall back to local in-memory numpy (the
default, unchanged behavior — see `vector_store.py`).

## Code verification sandbox

The orchestrator mounts `/var/run/docker.sock` so the code-sandbox
(`WarmDockerPool`) can spawn verification containers. If you don't use code
verification, remove that volume mount. The sandbox image defaults to
`vibe-thinker-sandbox:latest` — build it with:

```bash
docker build -f sandbox/Dockerfile -t vibe-thinker-sandbox:latest .
```

Egress from the sandbox is filtered by the Python SNI proxy (`sni-proxy`).
Edit the `RFSN_NETWORK_ALLOWLIST` environment variable (or the
`--network-allowlist` CLI flag) to allow the domains your code needs to reach.
The default allow-list is `pypi.org:443,files.pythonhosted.org:443`. The
proxy is hardened with a read-only filesystem, dropped capabilities, no new
privileges, non-root user, PID and memory limits, and a tmpfs `/tmp`.

The Envoy sidecar (`--envoy-sidecar`) is an alternative for standalone
transparent-routing experiments on the host and is not used in the compose
stack.

## Validation

Run the compose smoke test to verify the sidecars start and the hardened
`sni-proxy` correctly allows allowlisted HTTP/HTTPS traffic and blocks
non-allowlisted domains:

```bash
./scripts/test_compose.sh
```
