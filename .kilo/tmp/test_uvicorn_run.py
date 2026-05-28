"""Try to simulate what Docker ENTRYPOINT does: run uvicorn as a subprocess"""
import subprocess, sys, os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Simulate Docker: WORKDIR=/app, PYTHONPATH=/app, CMD ["uvicorn backend.app:app --port 8000"]
env = {
    **os.environ,
    "PYTHONPATH": "/app",
    "PORT": "8000"
}

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "--help"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd="/",
    env=env
)
out, err = proc.communicate(timeout=10)
print("Return code:", proc.returncode)
print("STDOUT:", out.decode()[:200])
print("STDERR:", err.decode()[:200])

# Now actually try to run uvicorn with the app specified
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd="/",
    env=env
)
out, err = proc.communicate(timeout=20)
print("\nUvicorn run return code:", proc.returncode)
print("STDOUT:", out.decode()[:300])
print("STDERR:", err.decode()[:1000])
