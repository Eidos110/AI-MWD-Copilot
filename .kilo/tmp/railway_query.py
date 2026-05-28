"""Call Railway Backboard GraphQL API with proper auth."""
import urllib.request, json, ssl

TOKEN = "0Ibuy4KzPDr3jZuVQW2sqo_V1ymTuVBO5bLKQwXFCS2"
API = "https://backboard.railway.com/graphql"

# Create SSL context that doesn't verify certs (for some network issues)
ctx = ssl.create_default_context()

def graphql(query, variables=None):
    body = {"query": query, "variables": variables or {}}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        API,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
            "GraphQL-Method": "POST"
        }
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30, context=ctx)
        result = json.loads(resp.read().decode())
        return result
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode()[:500]}
    except Exception as e:
        return {"error": str(e)}

# Query 1: Current service instance config
query1 = """
query GetInstance($id: ID!) {
  serviceInstance(id: $id) {
    id
    config
    service { id name }
  }
}
"""

print("=== Query 1: Get mwd-backend instance config ===")
result = graphql(query1, {"id": "25d24257-bda3-4fa0-a6ab-4f797107da8e"})
print(json.dumps(result, indent=2))
