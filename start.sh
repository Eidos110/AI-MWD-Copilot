#!/bin/bash
set -e

echo "Starting AI MWD Copilot..."

# Get port from environment or default to 8080
PORT=${PORT:-8080}
echo "Using port: $PORT"

# Start backend in background
cd /app
uvicorn backend.app:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
echo "Backend started on port 8000"

# Wait for backend to start
sleep 5
echo "Backend log:"
cat /tmp/backend.log

# Start frontend in background
cd /app/frontend
PORT=3000 node server.js > /tmp/frontend.log 2>&1 &
echo "Frontend started on port 3000"

# Wait for frontend to start
sleep 5
echo "Frontend log:"
cat /tmp/frontend.log

# Start nginx in foreground
nginx -g 'daemon off;'