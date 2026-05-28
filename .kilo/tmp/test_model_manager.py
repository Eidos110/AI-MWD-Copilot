"""Fast test: just import model_manager (without running it)"""
import subprocess, sys, os

backend_dir = os.path.dirname(os.path.abspath(__file__))

result = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, '.'); import backend.services.model_manager; print('OK')"],
    capture_output=True, text=True, timeout=60,
    cwd=backend_dir
)
print("Return code:", result.returncode)
print("STDOUT:", result.stdout[:300])
print("STDERR:", result.stderr[:1000])
