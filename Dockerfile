# =============================================================================
# vibe-thinker orchestrator — multi-stage Docker image
# =============================================================================
# Base: python:3.12-slim. Installs only the runtime deps the orchestrator needs
# (verified against the actual imports in hybrid_orchestrator.py, vibe_clr_async.py,
# rfsn_cli.py, federation_server.py, web/app.py, and requirements.txt).
#
# Build:
#   docker build -t vibe-thinker:latest .
# Run (see docker-compose.yml):
#   docker compose --profile cpu up
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder: install dependencies into a venv we can copy cleanly.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Build deps for packages that compile from source (numpy, scikit-learn,
# sentence-transformers pulls torch transitively — we keep the slim image
# and rely on pre-built wheels where possible).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtualenv so the final stage can copy just the installed packages.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies from the pinned requirements.txt.
# requirements.txt groups optional deps with comments; we install the full set
# so embeddings (numpy/sklearn/sentence-transformers), web (fastapi/uvicorn),
# federation (redis), and signing (cryptography) are all available. The
# orchestrator fail-closes gracefully when any optional dep is missing, but
# shipping them gives the full experience out of the box.
WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim image with just the venv + app code.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Runtime deps for code verification sandboxing (Docker CLI to talk to the
# mounted docker socket) and for wget-based healthchecks in compose.
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy the orchestrator's Python modules (the .py files at repo root) and the
# verifiers/ + sandbox/ + web/ subdirectories the orchestrator imports.
# .dockerignore excludes tests/, __pycache__, .git, *.gguf, and trajectory/
# memory JSON files so the image stays small.
COPY hybrid_orchestrator.py vibe_clr_async.py vibe_clr.py rfsn_cli.py \
     rfsn_job_queue.py federated_queue.py federation_server.py \
     persistent_cache.py bitemporal_log.py scoring.py math_solver.py \
     format_enforcer.py retrieval.py vector_store.py ruvllm_adapter.py \
     hardware_guardrail.py serialization.py signers.py web_security.py \
     vt_config.py vt_logging.py run_ui.py demo.py demo_v1.py \
     pyproject.toml ./
COPY verifiers/ ./verifiers/
COPY sandbox/ ./sandbox/
COPY web/ ./web/

# Persistent data directory for the audit log + trajectory store + CLR cache.
# docker-compose mounts a named volume here; when running standalone, the
# container writes to /data so it survives via a bind mount.
RUN mkdir -p /data
VOLUME ["/data"]

# Run as a non-root user for defense-in-depth (the orchestrator never needs
# root). The `vibe` user owns /app (read-only code) and /data (audit log +
# trajectory store + CLR cache). The /opt/venv stays root-owned and
# world-readable/executable so vibe can run python without owning the venv.
# Docker-socket access for the code sandbox is granted at run time via the
# mounted /var/run/docker.sock; if your socket is group-restricted, add the
# vibe user to the host's docker group (the GID varies by host).
RUN useradd -m -u 1000 vibe && chown -R vibe:vibe /app /data
USER vibe

# Default env vars consumed by rfsn_cli.py (overridable at run time).
# These match the env-var precedence in build_argparser (CLI > env > default).
ENV VIBE_THINKER_URL=http://127.0.0.1:8080 \
    GENERALIST_URL=http://127.0.0.1:8081 \
    AGENTDB_URL=http://agentdb:8088 \
    RFSN_PROXY_EGRESS=envoy:8888 \
    RFSN_AUDIT_LOG=/data/rfsn_jobs_bitemporal.jsonl \
    TRAJECTORY_STORE_PATH=/data/verified_trajectories.json \
    RFSN_MAX_CONCURRENT=2

# ENTRYPOINT is the orchestrator REPL. CMD is empty by default — rfsn_cli.py
# reads all flags from the VIBE_THINKER_* / RFSN_* env vars above, so no
# explicit args are needed. Override CMD to pass extra flags, e.g.
#   docker run vibe-thinker --max-concurrent 4 --no-clr
ENTRYPOINT ["python3", "rfsn_cli.py"]
CMD []
