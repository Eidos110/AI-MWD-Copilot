import urllib.request, json

urls = [
    "https://mwd-backend-production.up.railway.app/health",
    "https://mwd-backend-production.up.railway.app/api/v1/health",
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        print(f"GET {url} -> {resp.status}")
        print(resp.read().decode())
    except Exception as e:
        print(f"GET {url} -> ERROR: {e}")
    print()
