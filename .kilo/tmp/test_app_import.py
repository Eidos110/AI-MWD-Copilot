"""Import backend.app from the ROOT directory (simulating Docker WORKDIR=/app)"""
import subprocess, sys, os

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = {**os.environ, "PYTHONPATH": "/app"}

code = (
    "import sys; sys.path.insert(0, '/app'); "
    "from backend.app import app; "
    "print('OK:', app.title)"
)

proc = subprocess.Popen(
    [sys.executable, "-c", code],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=cwd, env=env
)

try:
    out, err = proc.communicate(timeout=30)
    print(f"Return code: {proc.returncode}")
    print(f"STDOUT:\n{out.decode()[:500]}")
    print(f"STDERR:\n{err.decode()[:2000]}")
except subprocess.TimeoutExpired:
    proc.kill()
    out, err = proc.communicate(timeout=5)
    print("TIMEOUT after 30s - uvicorn import likely blocked at model loading")
    print(f"STDOUT: {out.decode()[:200]}")
    print(f"STDERR: {err.decode()[:1000]}")
