#!/bin/bash
set -e

echo "Starting AI MWD Copilot..."

# Start backend
cd /app
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait a bit for backend to start
sleep 3

# Start frontend
cd /app/frontend
node server.js &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

# Wait for both
wait