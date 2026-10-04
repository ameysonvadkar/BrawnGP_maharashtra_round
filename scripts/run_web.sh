#!/usr/bin/env bash
# Start the Black Box web app: FastAPI backend (127.0.0.1:8000) + frontend (127.0.0.1:3000).
# Usage: ./scripts/run_web.sh        (Ctrl+C stops both)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies (first run)…"
  (cd frontend && npm install --ignore-scripts --no-audit --no-fund)
fi

"$PY" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

echo "Waiting for the API (first launch builds data/, a few seconds)…"
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/api/health >/dev/null && break
  sleep 1
done

echo "Black Box: http://127.0.0.1:3000"
cd frontend && npx vite dev
