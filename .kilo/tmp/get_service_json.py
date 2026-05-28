"""Use Railway CLI JSON output to get live carrier status."""
import subprocess, json, sys

def run_cmd(args, timeout=30):
    r = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-c"] + args,
        capture_output=True, text=True, timeout=timeout
    )
    stdout = r.stdout.strip()
    return stdout, r.stderr.strip()

# Get service status in JSON
stdout, stderr = run_cmd(["railway service status --json 2>&1"])
if stderr:
    print("STDERR:", stderr[:200])
if stdout:
    print("STDOUT:", stdout[:2000])
else:
    print("(no output)")
