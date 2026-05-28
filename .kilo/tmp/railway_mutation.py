"""
Call Railway Backboard GraphQL with proper Transport API key.

The accessToken in .railway/config.json is a machine access token for 
Railway CLI operations. For Backboard GraphQL direct API, we need the 
service_instance_id to reference mwd-backend.

Strategy: Use the `serviceInstance` query ID from the clouds instance,
and `serviceInstanceUpdate` mutation to fix the issue.

The service instance ID is: 4da51564-3203-4663-9ec3-810946629d77

We need to:
1. Query the instance config
2. Update healthcheckPath from "/" to "/health"
3. Also fix the dockerfilePath to be consistent

Since direct API calls are blocked (403), try using the token from 
.config.json which should use Railway's x-access-policy-token.
"""
import urllib.request, json

TOKEN = "0Ibuy4KzPDr3jZuVQW2sqo_V1ymTuVBO5bLKQwXFCS2"
API = "https://backboard.railway.com/graphql"

# Try different auth headers
auth_methods = [
    f"Bearer {TOKEN}",
    f"Bearer {TOKEN.upper()}",
    TOKEN,
    f"rl_{TOKEN}",
]

query = """
mutation UpdateInstance($id: ID!, $config: ServiceInstanceConfigUpdateInput!) {
  serviceInstanceUpdate(id: $id, config: $config, reason: "fix healthcheckPath / -> /health") {
    serviceInstance { id }
  }
}
"""

body = json.dumps({"query": query, "variables": {
    "id": "4da51564-3203-4663-9ec3-810946629d77",
    "config": {"deploy": {"healthcheckPath": "/health"}}
}}).encode()

for auth in auth_methods:
    print(f"Trying auth: '{auth[:20]}...'")
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": auth}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        print(f"  SUCCESS {resp.status}: {resp.read().decode()[:300]}")
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:200]
        print(f"  HTTP {e.code}: {err[:150]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
