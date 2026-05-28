import urllib.request, json

urls = [
    "https://mwd-backend-production.up.railway.app/health",
    "https://mwd-backend-production.up.railway.app/api/v1/health",
    "https://mwd-backend-production.up.railway.app/",
]

for url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"})
        resp = urllib.request.urlopen(req, timeout=15)
        print(f"GET {url} -> {resp.status}")
        print(resp.read().decode()[:300])
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        print(f"GET {url} -> {e.code} Err")
        print(body)
        # Check headers
        print("Headers:", dict(e.headers))
    except Exception as e:
        print(f"GET {url} -> ERROR: {e}")
    print("---")
