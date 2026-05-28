"""Build/ship/deploy lifecycle for 29c22d48 vs 46951e23."""
from datetime import datetime

ts1 = datetime(2026, 5, 19, 22, 36, 2)   # 29c22d48 DEPLOYING (UTC)
ts2 = datetime(2026, 5, 19, 22, 49, 0)   # now (approx UTC)

elapsed = (ts2 - ts1).total_seconds()
print(f"Time since 29c22d48 started: {elapsed:.0f}s ({elapsed/60:.1f}m)")
print(f"Build succeeded: Yes (image push showed in logs)")
print(f"Build failed at: carrier healthcheck (service unavailable 503)")
print(f"Health check path: /")
print(f"Health check timeout: 600s (10m per carrier config)")
print(f"Actual retries: ~18 at 10s intervals = 18*10=180s = 3m (not 10m)")
print()

# Conclusion:
print("=== KEY FINDING ===")
print("Building works. Deploying succeeds (uvicorn starts).")
print("Healthcheck at '/' fails even though '/' handler exists.")
print("The only way for carrier cloud to probe '/' and get 503 is if:")
print("  1. Uvicorn crashes immediately after start, OR")
print("  2. Uvicorn doesn't actually start")
print("  3. There's a cargo_cloud configuration issue")
print()
print("Build validation passes (saw Uvicorn running from loader.probe_docker_image)")
print("Deploy healthcheck fails → likely uvicorn crashes during carrier deploy phase")
print()
print("Let me check if mwd-backend carrrier image actually starts uvicorn correctly")
