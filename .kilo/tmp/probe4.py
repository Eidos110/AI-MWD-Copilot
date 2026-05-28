"""Probe carrier's healthcheck endpoint via edge.
Uses 2 probes: one at public-level and one carrier-close.
But we need to isolate the carrier behavior.
"""
import urllib.request, json, ssl, sys

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# TO-DIAGNOSE: Probe via private domain if exposed.
# RAILWAY_PRIVATE_DOMAIN from env says 'mwd-backend.railway.internal'
# But that's not accessible from outside Carrier's environment.
# However, we CAN try the PRIVATE_NETWORK_DOMAIN.
# Rail's internal carrier LANCarrier IP, then probe.

BACKEND = "mwd-backend-production.up.railway.app"
paths = ["/", "/health", "/docs", "/openapi.json"]

results = {}
for path in paths:
    url = f"https://{BACKEND}{path}"
    success = False
    for i in range(3):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Router/1.0", "Cache-Control": "no-cache"}
            )
            resp = urllib.request.urlopen(req, timeout=10, context=ctx)
            status = resp.status
            body = resp.read().decode()[:100]
            results[path] = f"HTTP {status}: {body[:50]}"
            success = True
            break
        except urllib.error.HTTPError as e:
            status = e.code
            body = e.read().decode()[:100]
            results[path] = f"HTTP {status}: {body[:50]} (fallback={e.headers.get('X-Railway-Fallback','?')})"
        except Exception as e:
            results[path] = f"ERR: {str(e)[:80]}"
        import time; time.sleep(0.3)

for path, result in results.items():
    print(f"  {path:20s} : {result}")
