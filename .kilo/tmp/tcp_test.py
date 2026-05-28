"""Basic connectivity test of deployed backend."""
import urllib.request, ssl

# Test if backend/service calls are handled by carrier
# Use a completely different port to rule-out conflict
BACKENDS = [
    "mwd-backend-production.up.railway.app",
]

# From the plan: `mwd-backend` service is PROBED AT `/` by carrier.
# Carrier returns http 404 (railway-edge fallback) from PROBE probe.
# Carrier seems to not mount validation on (edge-facing) service instance.
# Actually, as per Rail, the `railway status --json` shows
#   latestDeployment.Status = DEPLOYED for newest carrier-queues
#   latestDeployment.Status = FAILED if the cycle fails

# Let me analyze: The internal service carrier `29c22d48` is marked
# FAILED. What failed? Let me check: latestDeployment.Meta.reason = "deploy".
# Deploy.Reason = "deploy". Build.Strategy = "latest build".
# Deploy deploy.manifest = deploy manifest. Status = FAILED.
# Carrier deploy alert = "carrier_port_threshold or caller].
# Reinforce that the carrier signals SRV=?. Let me try without HTTP schema.

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for backend in BACKENDS:
    print(f"=== Testing {backend} ===")
    # Try a raw TCP socket test
    import socket
    hostname = backend
    try:
        sock = socket.create_connection((hostname, 443), timeout=5)
        print(f"  TCP 443: CONNECTED")
        # Send HTTP GET / HTTP/1.1
        sock.sendall(b"GET / HTTP/1.1\r\nHost: mwd-backend-production.up.railway.app\r\nConnection: close\r\nUser-Agent: python\r\n\r\n")
        data = b""
        while True:
            chunk = sock.recv(1024)
            if not chunk:
                break
            data += chunk
        print(f"  Raw HTTP response: {data.decode()[:200]}")
        sock.close()
    except Exception as e:
        print(f"  TCP error: {e}")
    
    print()

    # Also try HTTP
    try:
        req = urllib.request.Request(
            f"https://{backend}/",
            headers={"Host": backend}
        )
        resp = urllib.request.urlopen(req, timeout=10, context=ctx)
        print(f"  HTTPS / -> {resp.status} {resp.read()[:100]}")
    except urllib.error.HTTPError as e:
        print(f"  HTTPS / -> {e.code} {e.read()[:100]}")
    except Exception as e:
        print(f"  HTTPS error: {e}")
