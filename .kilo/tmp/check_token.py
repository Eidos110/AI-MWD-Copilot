"""Check Railway CLI internal config and find any usable token."""
import subprocess, json, os, sys

def get_config_content():
    with open(os.path.expanduser("~/.railway/config.json")) as f:
        return json.load(f)

config = get_config_content()
print("=== Auth Tokens in .railway/config.json ===")
user = config.get("user", {})
for key, val in user.items():
    if val and 'token' in key.lower():
        masked = val[:8] + "..." + val[-4:] if len(val) > 12 else "MASKED"
        print(f"  {key}: {masked}")

# Also check config file separately
print(f"\naccessToken: {user.get('accessToken', '')[:10]}...{user.get('accessToken', '')[-4:]}")
print(f"refreshToken: {user.get('refreshToken', '')[:10]}...{user.get('refreshToken', '')[-4:]}")
print(f"tokenExpiresAt: {user.get('tokenExpiresAt', 'N/A')}")

# Try "railway mcp" to see if we can access local token
import datetime
exp = user.get('tokenExpiresAt', 0)
if exp > 0:
    exp_dt = datetime.datetime.utcfromtimestamp(exp)
    print(f"Token expires: {exp_dt.isoformat()}")
    now = datetime.datetime.utcnow()
    print(f"Current time: {now.isoformat()}")
    expired = exp < now.timestamp()
    print(f"Token expired: {expired}")
