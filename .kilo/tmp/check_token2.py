"""Check Railway CLI internal config and find any usable token."""
import json, os, datetime

with open(os.path.expanduser("~/.railway/config.json")) as f:
    config = json.load(f)

user = config.get("user", {})
access_token = user.get("accessToken", "")
refresh_token = user.get("refreshToken", "")
token_expires = user.get("tokenExpiresAt", 0)

print(f"accessToken (masked): {access_token[:8]}...{access_token[-4:]}")
print(f"refreshToken (masked): {refresh_token[:10]}...{refresh_token[-4:]}")
if token_expires:
    exp_dt = datetime.datetime.utcfromtimestamp(token_expires)
    now = datetime.datetime.utcnow()
    print(f"Token expires: {exp_dt.isoformat()} UTC")
    print(f"Current UTC:  {now.isoformat()}")
    print(f"Expired: {token_expires < now.timestamp()}")
else:
    print("No tokenExpiresAt found")
