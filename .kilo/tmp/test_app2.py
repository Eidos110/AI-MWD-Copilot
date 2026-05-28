"""Try direct uvicorn test to check if / works"""
import subprocess, sys, os, time

# Try uvicorn from the repo root with the same Docker CMD
repo_root = r"E:\Code\Well-Logging-AI-AWD-Copilot-Deepseek"
env = {**os.environ, "PYTHONPATH": "/app"}

code = (
    "import sys; sys.path.insert(0, '/app'); "
    "from backend.app import app; "
    "print('APP_IMPORT_OK')"
)

# Quick test via Python import
proc = subprocess.Popen(
    [sys.executable, "-c", code],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=repo_root,
    timeout=20
)
try:
    out, err = proc.communicate(timeout=30)
    print("Import: RC", proc.returncode)
    print("STDOUT:", out.decode()[:300])
    print("STDERR:", err.decode()[:2000])
except subprocess.TimeoutExpired:
    proc.kill()
    out, err = proc.communicate(timeout=3)
    print("Import TIMEOUT - possibly blocking at model load")
    print("STDOUT:", out.decode()[:200])
    print("STDERR:", err.decode()[:1000])

# Also try without model loading
code2 = (
    "import sys; sys.path.insert(0, '/app'); "
    "from backend.core.config import settings; "
    "print('CONFIG_OK:', settings.APP_NAME)"
)
proc2 = subprocess.Popen(
    [sys.executable, "-c", code2],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=repo_root, timeout=10
)
try:
    out, err = proc2.communicate(timeout=15)
    print("\nConfig: RC", proc2.returncode)
    print("STDOUT:", out.decode()[:300])
    print("STDERR:", err.decode()[:1000])
except subprocess.TimeoutExpired:
    proc2.kill()
    out, err = proc2.communicate(timeout=3)
    print("\nConfig TIMEOUT")
