"""Query Railway Backboard with graphql endpoint directly using token verification."""
import urllib.request, json, ssl

# Try the verified correct endpoint and auth
api_url = "https://backboard.railway.com/graphql"
token = "0Ibuy4KzPDr3jZuVQW2sqo_V1ymTuVBO5bLKQwXFCS2"

query = """
query IntrospectionQuery {
  __schema {
    types { name }
    queryType { name fields { name } }
  }
}
"""

body = json.dumps({"query": query}).encode()
ctx = ssl.create_default_context()

req = urllib.request.Request(
    api_url,
    data=body,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
)

try:
    resp = urllib.request.urlopen(req, timeout=30, context=ctx)
    print("SUCCESS", resp.status)
    data = json.loads(resp.read().decode())
    # Print just the schema types
    if "data" in data and "__schema" in data["data"]:
        types = data["data"]["__schema"]["types"]
        print(f"Schema returned {len(types)} types")
        print(", ".join([t["name"] for t in types[:20]]))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}")
    body_out = e.read().decode()[:500]
    print(body_out)
except Exception as e:
    print(f"Error: {e}")
