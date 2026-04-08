#!/bin/bash
set -e

# Start backend in background
cd /app
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 5

# Start frontend
cd /app/frontend
node server.js &

# Wait for both
wait $BACKEND_PID