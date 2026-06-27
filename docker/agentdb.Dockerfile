# =============================================================================
# AgentDB sidecar — PLACEHOLDER Dockerfile
# =============================================================================
# RuFlo/AgentDB is the vector store sidecar from the ruvnet/ruflo project
# (https://github.com/ruvnet/ruflo). The orchestrator calls it via
#   POST /v1/vector/search   (vector_store.py:405)
# on the conventional port 8088 (rfsn_cli.py:540, :882).
#
# IMPORTANT: There is NO published AgentDB Docker image on Docker Hub as of
# writing. This Dockerfile is a PLACEHOLDER so `docker compose build` does not
# fail with a missing-file error before you have a chance to supply the real
# binary. It builds a trivial image that prints a message and exits.
#
# To make this service actually serve vector searches, do ONE of:
#   1. Build a real AgentDB image from the ruvnet/ruflo source and set
#      `image:` in docker-compose.yml (replacing the `build:` block).
#   2. Drop a real AgentDB binary into this docker/ directory and replace
#      the COPY/RUN below to install + run it on port 8088.
#   3. Point AGENTDB_URL at a host-side AgentDB instance and remove the
#      `agentdb` service from docker-compose.yml entirely.
#
# Until you do one of the above, the orchestrator fail-closes:
# AgentDBVectorStore returns [] when the sidecar is down (vector_store.py:46-57)
# and reads fall back to local in-memory numpy (the default, unchanged behavior).
# =============================================================================

FROM python:3.12-slim

# Placeholder: a tiny HTTP server on port 8088 that always returns an empty
# search result. This lets the compose stack start and the healthcheck pass,
# while making it obvious that no real vector search is happening.
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# Replace this block with the real AgentDB binary, e.g.:
#   COPY agentdb /usr/local/bin/agentdb
#   RUN chmod +x /usr/local/bin/agentdb
#   EXPOSE 8088
#   ENTRYPOINT ["/usr/local/bin/agentdb", "--host", "0.0.0.0", "--port", "8088"]
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8088

# Placeholder entrypoint: a 12-line Python HTTP server returning empty results
# for POST /v1/vector/search and 200 for GET /health. Replace with the real
# AgentDB binary (see comment above).
RUN printf '%s\n' \
    'import http.server, json' \
    'class H(http.server.BaseHTTPRequestHandler):' \
    '    def do_GET(self):' \
    '        if self.path == "/health":' \
    '            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")' \
    '        else: self.send_response(404); self.end_headers()' \
    '    def do_POST(self):' \
    '        if self.path == "/v1/vector/search":' \
    '            self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()' \
    '            self.wfile.write(json.dumps({"results":[]}).encode())' \
    '        else: self.send_response(404); self.end_headers()' \
    '    def log_message(self, *a): pass' \
    'http.server.HTTPServer(("0.0.0.0",8088),H).serve_forever()' \
    > /usr/local/bin/agentdb_placeholder.py

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=10s \
    CMD wget -q -O- http://127.0.0.1:8088/health || exit 1

ENTRYPOINT ["python3", "/usr/local/bin/agentdb_placeholder.py"]
