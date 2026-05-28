import urllib.request, json, sys

TOKEN = "0Ibuy4KzPDr3jZuVQW2sqo_V1ymTuVBO5bLKQwXFCS2"
API = "https://backboard.railway.com/graphql"

def graphql(query, variables=None):
    body = {"query": query}
    if variables:
        body["variables"] = variables
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        API,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}"
        }
    )
    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read().decode())
        print(json.dumps(result, indent=2))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")

# Phase 1: Check mwd-backend instance
print("=== Phase 1: mwd-backend Instance Query ===")
graphql(
    '{ serviceInstance(id: "25d24257-bda3-4fa0-a6ab-4f797107da8e") { id service { id name } config } }'
)
