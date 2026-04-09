#!/bin/bash
set -e

echo "Starting AI MWD Copilot..."

# Start backend in background
cd /app
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
echo "Backend started on port 8000"

# Wait for backend to start
sleep 3

# Start frontend in background
cd /app/frontend
node server.js &
echo "Frontend started on port 3000"

# Wait for frontend to start
sleep 3

# Start nginx in foreground (this keeps container running)
nginx -g 'daemon off;'