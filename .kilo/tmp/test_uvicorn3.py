"""Run uvicorn and capture full output"""
import subprocess, sys, os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env = {
    **os.environ,
    "PYTHONPATH": "/app"
}

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=repo_root,
    env=env,
    close_fds=True
)

# Give time for startup errors
import time
time.sleep(3)

out, err = proc.communicate(timeout=5)
print("Return code:", proc.returncode)
print("STDOUT:", out.decode()[:3000])
print("STDERR:", err.decode()[:3000])
