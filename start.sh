#!/bin/bash
set -e

echo "Starting AI MWD Copilot..."

# Read port from Railway's injected $PORT, default to 8000
PORT=${PORT:-8000}
echo "Using port: $PORT"

# Start backend in foreground on theRailway-assigned port
cd /app
echo "Starting FastAPI backend on port $PORT..."
exec uvicorn backend.app:app --host 0.0.0.0 --port "$PORT"
