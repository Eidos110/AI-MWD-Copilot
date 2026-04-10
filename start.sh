#!/bin/bash
set -e

echo "Starting AI MWD Copilot..."

# Get port from environment or default to 8080
PORT=${PORT:-8080}
echo "Using port: $PORT"

# Start backend in foreground on Railway port
cd /app
echo "Starting simple Python HTTP server on port $PORT..."
python3 backend/app.py