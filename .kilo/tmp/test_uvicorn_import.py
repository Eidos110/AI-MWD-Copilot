"""Debug: Try to start uvicorn directly to see what error it gives.
This mimics what carrier would do: CMD ["uvicorn", "backend.app:app", "--port","8000"] from /app
"""
import subprocess, sys, os

backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)

result = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, '.'); from backend import app; print('SUCCESS')"],
    capture_output=True, text=True, timeout=30, env={**os.environ, "PYTHONPATH": "."}
)
print("Return code:", result.returncode)
print("STDOUT:", result.stdout[:500])
print("STDERR:", result.stderr[:1000])
