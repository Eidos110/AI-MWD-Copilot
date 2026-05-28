"""Use Railway CLI token to call Backboard API."""
import subprocess, json, os

TOKEN = "0Ibuy4KzPDr3jZuVQW2sqo_V1ymTuVBO5bLKQwXFCS2"
API = "https://backboard.railway.com/graphql"

query = """
query GetInstance($id: ID!) {
  serviceInstance(id: $id) {
    id
    config
    service { id name }
  }
}
"""

# Use curl.exe directly
body = {"query": query, "variables": {"id": "25d24257-bda3-4fa0-a6ab-4f797107da8e"}}
data = json.dumps(body)

import urllib.request, ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request(
    API,
    data=data.encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    }
)

try:
    resp = urllib.request.urlopen(req, timeout=30, context=ctx)
    print("SUCCESS:", resp.read().decode()[:1000])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:500]}")
except Exception as e:
    print(f"ERROR: {e}")
