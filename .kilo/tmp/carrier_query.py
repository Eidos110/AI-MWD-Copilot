"""Try Railway Carrier Send API with different auth methods."""
import urllib.request, json, ssl, sys

URL = "https://backboard.railway.com/graphql"
TOKEN = "0Ibuy4KzPDr3jZuVQW2sqo_V1ymTuVBO5bLKQwXFCS2"

# Build a valid mutation query
mutation = """
mutation UpdateInstance($id: ID!, $config: ServiceInstanceConfigUpdateInput!) {
  serviceInstanceUpdate(id: $id, config: $config) {
    serviceInstance { id config { deploy { healthcheckPath } } }
  }
}
"""

body = json.dumps({"query": mutation, "variables": {
    "id": "4da51564-3203-4663-9ec3-810946629d77",
    "config": {"deploy": {"healthcheckPath": "/health"}}
}}).encode()

# Try multiple auth headers
headers_list = [
    {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    {"Content-Type": "application/json", "Authorization": TOKEN},
    {"Content-Type": "application/json", "X-Access-Token": TOKEN},
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for i, hdrs in enumerate(headers_list, 1):
    req = urllib.request.Request(URL, data=body, headers=hdrs)
    try:
        resp = urllib.request.urlopen(req, timeout=15, context=ctx)
        data = resp.read().decode()[:500]
        print(f"Try {i} [{list(hdrs.keys())[1]}]: SUCCESS {resp.status}")
        print(f"  {data[:300]}")
        break
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:300]
        print(f"Try {i} [{list(hdrs.keys())[-1]}]: HTTP {e.code}: {err[:150]}")
    except Exception as e:
        print(f"Try {i}: ERROR: {e}")
    print()

# Also try Query to see if any auth works at all
query = "{ viewer { id username } }"
query_body = json.dumps({"query": query}).encode()
req2 = urllib.request.Request(URL, data=query_body, headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
})
try:
    resp2 = urllib.request.urlopen(req2, timeout=15, context=ctx)
    data2 = resp2.read().decode()[:300]
    print(f"\nQuery test: {resp2.status} - {data2[:200]}")
except urllib.error.HTTPError as e:
    print(f"\nQuery test: HTTP {e.code} - {e.read().decode()[:150]}")
except Exception as e:
    print(f"\nQuery test: ERROR - {e}")
