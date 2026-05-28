"""Try Railway Backboard API with the CURRENT valid token."""
import urllib.request, json, ssl

TOKEN = "ng81a4utPayqEx8DSW7rO0OoNQ6gPjt3XJE_gNTbuIJ"
API = "https://backboard.railway.com/graphql"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Test query first
query = """
mutation UpdateInstance($id: ID!, $config: ServiceInstanceConfigUpdateInput!) {
  serviceInstanceUpdate(id: $id, config: $config) {
    serviceInstance {
      id
      config {
        deploy {
          healthcheckPath
        }
      }
    }
  }
}
"""

body = json.dumps({"query": query, "variables": {
    "id": "4da51564-3203-4663-9ec3-810946629d77",
    "config": {"deploy": {"healthcheckPath": "/health"}}
}}).encode()

req = urllib.request.Request(
    API,
    data=body,
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
)

try:
    resp = urllib.request.urlopen(req, timeout=30, context=ctx)
    data = json.loads(resp.read().decode())
    if "errors" in data:
        print(f"ERROR in response: {json.dumps(data['errors'], indent=2)[:500]}")
    elif "data" in data:
        print("SUCCESS:", json.dumps(data["data"], indent=2)[:500])
    else:
        print(f"Unknown response: {json.dumps(data)[:500]}")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:500]}")
except Exception as e:
    print(f"ERROR: {e}")
