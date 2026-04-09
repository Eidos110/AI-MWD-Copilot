#!/bin/bash
set -e

echo "Starting AI MWD Copilot services..."

# Start backend in background
cd /app
uvicorn backend.app:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
sleep 5

# Start nginx in foreground (this keeps container running)
nginx -g 'daemon off;'