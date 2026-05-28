"""Analyze deployment build and healthcheck failure patterns."""
from datetime import datetime

data = {
    "29c22d48": "2026-05-19 22:36:02",
    "fcd890be": "2026-05-19 21:53:26",
    "46951e23": "2026-05-19 21:53:17",
    "d74b4cde": "2026-05-19 21:05:31",
    "986130a9": "2026-05-19 20:47:42",
}

# All 22 deployments - check time per deploy
print("Deployments ordered逆行chronologically (most recent first):")
for i, (name, ts) in enumerate(data.items(), 1):
    t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    if i < len(data):
        next_name, next_ts = list(data.items())[i]
        next_t = datetime.strptime(next_ts, "%Y-%m-%d %H:%M:%S")
        gap = (t - next_t).total_seconds()
    else:
        gap = None
    print(f"  {i}. {name[:10]} @ {ts} {'(first deploy)' if gap is None else f'gap: {gap:.0f}s from next'}")

print(f"\n  Newest deploy 29c22d48 is at {data['29c22d48']}")
print(f"  Previous fcd890be is at {data['fcd890be']}")
print(f"  Gap between them: {(datetime.strptime(data['fcd890be'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(data['29c22d48'], '%Y-%m-%d %H:%M:%S')).total_seconds():.0f}s")

print(f"\n  fcd890be vs 46951e23 gap: {(datetime.strptime(data['fcd890be'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(data['46951e23'], '%Y-%m-%d %H:%M:%S')).total_seconds():.0f}s (9s from plan)")
