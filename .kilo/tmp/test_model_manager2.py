"""Test model_manager import with proper PYTHONPATH"""
import subprocess, sys, os

backend_dir = os.path.dirname(os.path.abspath(__file__))

# Docker runs from /app with PYTHONPATH=/app
env = {**os.environ, "PYTHONPATH": backend_dir}

result = subprocess.run(
    [sys.executable, "-c",
     "import backend.services.model_manager; print('OK')"],
    capture_output=True, text=True, timeout=60, env=env
)
print("Return code:", result.returncode)
print("STDOUT:", result.stdout[:300])
print("STDERR:", result.stderr[:2000])
