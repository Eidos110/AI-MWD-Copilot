"""Determine live failure pattern by analyzing build timestamps."""
import json

# All deployments in order (latest first)
deployments = """
fcd890be-3bc0-4e19-961d-8429ca8945ed | FAILED | 2026-05-19 21:53:26
46951e23-e169-4ef9-a508-38add48be67c | FAILED | 2026-05-19 21:53:17
d74b4cde-35a4-461f-a3ac-b1c7e28938cd | FAILED | 2026-05-19 21:05:31
986130a9-76a6-4b8c-880b-5daa44a8821d | FAILED | 2026-05-19 20:47:42
f733a359-47dc-4354-8501-f26c9548352e | FAILED | 2026-05-19 20:45:50
34e360a6-25b2-428e-939b-a2c1e9741eb0 | FAILED | 2026-05-19 20:38:01
348d6656-1d82-445d-a0dd-8df5830b3fc7 | FAILED | 2026-05-19 20:28:45
a4d3ef82-5c7f-4e45-8b45-4180952a9af6 | FAILED | 2026-05-19 20:02:52
7a2b4560-75d0-4b34-a393-a54abb0b6451 | FAILED | 2026-05-19 19:59:00
a7f9a85a-5e07-4dfd-b600-a558a2dbe1f1 | FAILED | 2026-05-19 19:43:58
93f46ce1-d8e2-4fe5-8b82-cf680138165b | FAILED | 2026-05-19 19:21:41
08862719-c70a-4180-b16d-74c634408e71 | FAILED | 2026-05-19 19:08:12
0b1a031b-9727-45a1-bcca-104783b0c38a | FAILED | 2026-05-19 18:50:59
6a71187e-a342-4763-a2a4-86483e88d648 | FAILED | 2026-05-19 18:16:41
69badc9e-e504-4589-bdbc-6cec1e01426f | FAILED | 2026-05-19 18:09:45
655c1e9d-42a3-4784-be2d-c4dcdf89b164 | FAILED | 2026-05-19 17:57:26
a73a525a-2f8d-42c8-a5c1-12dbc83d4765 | FAILED | 2026-05-19 17:57:08
983b21d3-e631-4a7a-8723-d3483c6ca518 | FAILED | 2026-05-19 17:54:10
0b5c05f5-2db9-4af5-877e-f419882e363d | FAILED | 2026-05-19 17:44:34
bce2974c-efba-4d3f-b08a-5bb58ee605e7 | FAILED | 2026-05-19 17:44:22
"""

lines = [l.strip() for l in deployments.strip().split('\n') if l.strip()]
print(f"Total failed deploys: {len(lines)}")
for i, line in enumerate(lines[:5]):
    parts = line.split(' | ')
    name = parts[0][:12]
    ts = parts[-1]
    print(f"  #{i+1}: {name}... @ {ts}")

# Time between consecutive deploys
import datetime
times = []
for line in lines:
    parts = line.split(' | ')
    ts_str = ' '.join(parts[-1].split()[:2])
    try:
        t = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        times.append(t)
    except: pass

print("\nTime gaps between deploys:")
for i in range(1, len(times)):
    gap = (times[i-1] - times[i]).total_seconds()
    print(f"  {times[i].strftime('%H:%M:%S')} gap from prev: {gap:.0f}s ({gap/60:.1f}m)")
