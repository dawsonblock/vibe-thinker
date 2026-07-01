#!/usr/bin/env bash
# Compose smoke test — proves the CPU compose stack sidecars start and the
# hardened sni-proxy service works as an HTTP/HTTPS CONNECT proxy.
#
# Starts only the shared sidecars (redis + sni-proxy) with an explicit
# allow-list for the test domains, then exercises allowlisted HTTP,
# allowlisted HTTPS, and a blocked domain. Does NOT start the orchestrator
# or any LLM inference server, so no model is required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[compose] validating compose config..."
docker compose config >/dev/null

echo "[compose] starting sidecars with test allow-list..."
RFSN_NETWORK_ALLOWLIST=example.com:80,example.com:443 \
  docker compose up -d redis sni-proxy

cleanup() {
  echo "[compose] cleaning up..."
  docker compose down --remove-orphans
}
trap cleanup EXIT

echo "[compose] waiting for sni-proxy to be healthy..."
for i in {1..30}; do
  if docker compose ps sni-proxy | grep -q "healthy"; then
    break
  fi
  sleep 2
done

docker compose ps

echo "[compose] testing allowlisted HTTP through proxy..."
docker run --rm \
  --network "$(basename "$ROOT")_default" \
  -e HTTP_PROXY=http://sni-proxy:8888 \
  -e HTTPS_PROXY=http://sni-proxy:8888 \
  python:3.12-slim \
  python3 - <<'PY'
import urllib.request
r = urllib.request.urlopen("http://example.com", timeout=15)
print("HTTP_STATUS", r.status)
PY

echo "[compose] testing allowlisted HTTPS through proxy..."
docker run --rm \
  --network "$(basename "$ROOT")_default" \
  -e HTTP_PROXY=http://sni-proxy:8888 \
  -e HTTPS_PROXY=http://sni-proxy:8888 \
  python:3.12-slim \
  python3 - <<'PY'
import ssl, urllib.request
ctx = ssl.create_default_context()
r = urllib.request.urlopen("https://example.com", timeout=15, context=ctx)
print("HTTPS_STATUS", r.status)
PY

echo "[compose] testing blocked domain..."
docker run --rm \
  --network "$(basename "$ROOT")_default" \
  -e HTTP_PROXY=http://sni-proxy:8888 \
  -e HTTPS_PROXY=http://sni-proxy:8888 \
  python:3.12-slim \
  python3 - <<'PY'
import urllib.request
try:
    urllib.request.urlopen("http://httpbin.org/get", timeout=10)
except Exception as e:
    print("BLOCKED", type(e).__name__)
else:
    raise SystemExit("expected blocked domain to fail")
PY

echo "Compose smoke PASSED."
