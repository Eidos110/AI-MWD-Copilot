"""Test FastAPI app import"""
import subprocess, sys, os

repo_root = r"E:\Code\Well-Logging-AI-AWD-Copilot-Deepseek"

code = (
    "import sys; sys.path.insert(0, '/app'); "
    "from backend.app import app; "
    "print('APP_IMPORT_OK')"
)

proc = subprocess.Popen(
    [sys.executable, "-c", code],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=repo_root,
    env={**os.environ, "PYTHONPATH": "/app"}
)
try:
    out, err = proc.communicate(timeout=30)
    print("Import RC:", proc.returncode)
    print("STDOUT:", out.decode()[:500])
    print("STDERR:", err.decode()[:2000])
except subprocess.TimeoutExpired:
    proc.kill()
    out, err = proc.communicate()
    print("Import TIMEOUT - possibly blocked")
    print("STDOUT:", out.decode()[:200])
    print("STDERR:", err.decode()[:1000])
