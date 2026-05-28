"""Probe the running backend service from multiple angles."""
import urllib.request, json, socket

BACKEND_URL = "https://mwd-backend-production.up.railway.app"

print("=== Testing backend endpoints ===\n")

endpoints = [
    "/",
    "/health",
    "/api/v1/health",
    "/docs",
    "/openapi.json",
]

for path in endpoints:
    url = f"{BACKEND_URL}{path}"
    try:
        req = urllib.request.Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "curl/7.88"})
        resp = urllib.request.urlopen(req, timeout=15)
        body = resp.read().decode()[:200]
        print(f"GET {path} -> {resp.status} {resp.reason}")
        print(f"  Body: {body[:150]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:150]
        print(f"GET {path} -> {e.code} {e.reason}")
        print(f"  Body: {body[:100]}")
        print(f"  Server: {e.headers.get('Server','?')}")
        print(f"  X-Railway-Fallback: {e.headers.get('X-Railway-Fallback','?')}")
    except Exception as e:
        print(f"GET {path} -> ERROR: {type(e).__name__}: {e}")
    print()

# Also check TCP socket
print("=== TCP Connection Test ===")
try:
    s = socket.create_connection(("mwd-backend-production.up.railway.app", 443), timeout=5)
    print(f"TCP 443 on mwd-backend-production.up.railway.app: OPEN")
    s.close()
except Exception as e:
    print(f"TCP connection: {e}")
