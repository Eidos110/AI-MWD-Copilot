"""Run uvicorn from DOCKER WORKDIR (=/app) with PYTHONPATH=/app"""
import subprocess, sys, os, stat

# Reproduce the Docker CMD exactly:
# CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
# WORKDIR /app, PYTHONPATH=/app

repo_root = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(repo_root)

env = {
    **os.environ,
    "PYTHONPATH": "/app",
    "PORT": "8000"
}

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=repo_root,
    env=env,
    close_fds=True
)

# Kill after 10 seconds to get the startup output
import time
time.sleep(8)

# Send SIGTERM to uvicorn
proc.terminate()
out, err = proc.communicate(timeout=5)
print("Return code:", proc.returncode)
print("STDOUT:", out.decode()[:2000])
print("STDERR:", err.decode()[:2000])
