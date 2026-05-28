"""Set Railway environment variables for mwd-backend والتي."""
import subprocess, json

# Set MWD_CORS_ORIGINS
r1 = subprocess.run(
    ["powershell", "-ExecutionPolicy", "Bypass", "-c",
     "railway variable set MWD_CORS_ORIGINS='[\"*\"]' -s mwd-backend 2>&1"],
    capture_output=True, text=True, timeout=30
)
print("Set MWD_CORS_ORIGINS:", r1.stdout.strip(), r1.stderr.strip()[:200])

# Set NEXT_PUBLIC_API_URL for frontend
r2 = subprocess.run(
    ["powershell", "-ExecutionPolicy", "Bypass", "-c",
     "railway variable set NEXT_PUBLIC_API_URL=https://mwd-backend-production.up.railway.app -s mwd-frontend 2>&1"],
    capture_output=True, text=True, timeout=30
)
print("Set NEXT_PUBLIC_API_URL:", r2.stdout.strip(), r2.stderr.strip()[:200])
