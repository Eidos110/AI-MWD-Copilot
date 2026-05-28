"""Probe the actual live backend endpoints more carefully."""
import urllib.request, json, ssl, socket, time

BACKEND = "mwd-backend-production.up.railway.app"
ctx = ssl.create_default_context()

print(f"=== Testing {BACKEND} ===\n")

# Test specific endpoint
paths = ["/", "/health", "/api/v1/health", "/docs"]
for path in paths:
    for attempt in range(2):
        url = f"https://{BACKEND}{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8", "Connection": "close"})
            resp = urllib.request.urlopen(req, timeout=10, context=ctx)
            print(f"  {path} -> {resp.status} {resp.read().decode()[:100]}")
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:100]
            headers = dict(e.headers)
            print(f"  {path} -> {e.code} | headers: {{server: {headers.get('Server','?')}, fallback: {headers.get('X-Railway-Fallback','?')}}} | body: {body[:60]}")
        except socket.timeout:
            print(f"  {path} -> TCP TIMEOUT")
        except Exception as e:
            print(f"  {path} -> {e}")
    time.sleep(0.5)
