"""
Test what the actual deployed /health endpoint behavior is.
Also check if FastAPI /docs works.
"""
import urllib.request, json, time

BACKEND = "mwd-backend-production.up.railway.app"
ctx = None  # default SSL

# Test what Docker shows:
# Pick specific probe path
paths = ["/", "/docs", "/openapi.json", "/health"]
for path in paths:
    url = f"https://{BACKEND}{path}"
    for i in range(1):
        try:
            req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
            resp = urllib.request.urlopen(req, timeout=10)
            body = resp.read().decode()[:150]
            print(f"GET {path} -> {resp.status}: {body[:80]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:150]
            print(f"GET {path} -> {e.code}: {body[:80]}")
            print(f"  headers: fallback={e.headers.get('X-Railway-Fallback','none')}, server={e.headers.get('Server','?')}")
        except Exception as e:
            print(f"GET {path} -> ERR: {e}")
    time.sleep(0.3)
    print()
