#!/bin/bash
set -e

echo "Starting AI MWD Copilot..."

# Get port from environment or default to 8080
PORT=${PORT:-8080}
echo "Using port: $PORT"
echo "Environment variables:"
env | grep -E "(PORT|MWD_)" || echo "No MWD_ or PORT env vars found"

# Start backend in foreground on Railway port
cd /app
echo "Starting backend on port $PORT..."
echo "Command: uvicorn backend.app:app --host 0.0.0.0 --port $PORT"
exec uvicorn backend.app:app --host 0.0.0.0 --port $PORT