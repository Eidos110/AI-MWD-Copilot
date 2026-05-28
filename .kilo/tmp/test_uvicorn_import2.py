"""Test app import from repo root (simulating Docker WORKDIR /app)"""
import subprocess, sys, os

# Simulate Docker: from /app, with PYTHONPATH=/app,
# CMD=["uvicorn", "backend.app:app", "--port", "8000"]
# That's the same as running uvicorn backend.app:app (uvicorn adds PYTHONPATH=/app if needed)
backend_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(backend_dir)

env = {**os.environ, "PYTHONPATH": "/app"}

result = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, '/app'); from backend.app import app; print('OK: uvicorn app loaded')"],
    capture_output=True, text=True, timeout=30, env=env
)
print("Return code:", result.returncode)
print("STDOUT:", result.stdout[:500])
print("STDERR:", result.stderr[:1000])
