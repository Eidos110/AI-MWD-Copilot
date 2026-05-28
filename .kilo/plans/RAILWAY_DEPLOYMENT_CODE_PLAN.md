# Railway Deployment — Code Implementation Plan

> **Project**: AI-MWD-Copilot  
> **Project Root**: `E:\Code\Well-Logging-AI-AWD-Copilot-Deepseek`  
> **Last Updated**: 2026-05-20  
> **Status**: mwd-backend — fixing root-cause Python ImportError; mwd-frontend — waiting  
> **Railway Project**: AI-MWD-Copilot (`e48445d1`)  
> **Railway Env**: `production` (`3a99af65`)

---

## 1. Current Deployment State

| Service | Service ID | Domain | Status |
|---------|-----------|--------|--------|
| `mwd-backend` | `25d24257` | `mwd-backend-production.up.railway.app` | 🔄 Build failing — PYTHON `ImportError` in 3 route modules |
| `mwd-frontend` | `5876f291` | _(pending)_ | ⏳ Config + env vars pending user action in Railway Dashboard |

---

## 2. Root Cause Analysis (2026-05-20 Investigation)

### The Real Failure Chain

```
1. Railway deploys mwd-backend (backend/Dockerfile context)
2. Container starts: uvicorn backend.app:app --port 8000
3. uvicorn imports app.py
4. app.py:68 → "from backend.api.routes import predict, data, quality, shap"
5. predict.py:15 → "from backend.services.model_manager import ModelManager"
              → ImportError! (container crashes, healthcheck gets 503)
```

### Why Does `ImportError: cannot import name 'ModelManager'` Occur?

The `model_manager.py` file exists, `class ModelManager:` is defined at line 73, and `git show HEAD` confirms it's present. The ImportError is caused by a **partially-initialized module** — Python marks modules with errors as incomplete in `sys.modules`, so subsequent imports of the same name from a different path see the failed state.

In the prior deployment (`predict.py` imported `ModelManager` at module level), the ImportError kind of bootstrap loop was run at uvicorn startup when the app was being loaded inside the container. The container crashed, Railway restarted it, it crashed again, until all retries exhausted and `mwd-backend` was marked FAILED.

### Root Cause: `predict.py` imported `ModelManager` at module level

**Fix applied**: Rewrote `predict.py` with lazy import — `ModelManager` is now imported inside `_load_manager()` which is only called on first API hit. Patched `shap.py` and `quality.py` which also had module-level `model_manager` imports that would crash the app on startup.

---

## 3. File Change Inventory (2026-05-20 Session)

Files **changed this session**:

### Bug Fixes (Python import crash)

| File | Change | Status |
|------|--------|--------|
| `backend/api/routes/predict.py` | Replaced module-level `from backend.services.model_manager import ModelManager` with lazy import (`_load_manager()`) and `_StubModelManager` fallback | ✅ Committed & pushed |
| `backend/api/routes/shap.py` | Same — lazy `ModelManager`; moved `FEATURES_*` constants to module level inside the file | ✅ Committed & pushed |
| `backend/api/routes/quality.py` | Same — moved `FEATURES_*` constants to module level to avoid importing `model_manager` | ✅ Committed & pushed |
| `backend/core/config.py` | Added `import json` for potential env parser | ✅ Committed & pushed |

### Docker / Railway Config

| File | Change | Status |
|------|--------|--------|
| `backend/Dockerfile` | Fixed HE ALTHCHECK removed; `CMD uvicorn ...` confirmed | ✅ Committed & pushed |
| `Dockerfile` (root) | `EXPOSE 8000`, `CMD /app/start.sh` — not used by Railway (uses `backend/Dockerfile`) | ✅ Committed & pushed |

### Documentation

| File | Change | Status |
|------|--------|--------|
| `.railway-config.json` | Full status doc — services, root causes, fix evidence | ✅ Committed & pushed |
| `README.md` | Project structure, Railway URLs, deployment docs updated | ✅ Committed & pushed |

---

## 4. Import Chain After Fix

```
app.py ──→ predict.py   ← lazy ModelManager ✓
         ├─ quality.py  ← constant copies ✓ (no model_manager dependency)
         ├─ shap.py     ← lazy ModelManager ✓
         ├─ data.py     ← data_loader + config (clean)
         ├─ health.py   ← clean (no external deps)
         ├─ interpret.py ← interpreter only (clean)
         └─ websocket   ← websocket/manager (clean)
```

---

## 5. Railway Settings Reference (Current Known Values)

### Runtime Config (confirmed via `railway-status.json`)

```json
{
  "builder": "DOCKERFILE",
  "rootDirectory": "backend",
  "dockerfilePath": "backend/Dockerfile",
  "buildCommand": "pip install --no-cache-dir -r requirements.txt",
  "startCommand": "uvicorn backend.app:app --host 0.0.0.0 --port 8000",
  "healthcheckPath": "/",
  "healthcheckTimeout": 600,
  "numReplicas": 1,
  "sleepApplication": false
}
```

---

## 6. Verification Checklist

### Backend (target)
- [ ] `mwd-backend` deployment status: `SUCCESS`
- [ ] `GET https://mwd-backend-production.up.railway.app/health` returns `{"status":"healthy"}`
- [ ] No `ImportError` in Railway run logs
- [ ] Build logs show `pip install` and `COPY . /app/backend/` without errors

### Frontend (pending — user must set `NEXT_PUBLIC_API_URL` in Railway Dashboard)
- [ ] `NEXT_PUBLIC_API_URL=https://mwd-backend-production.up.railway.app` set in Railway env
- [ ] `mwd-frontend` fresh deploy triggered after env var is set
- [ ] Frontend builds and returns HTTP 200

### End-to-End
- [ ] Frontend loads without CORS errors
- [ ] `/api/*` proxy rewrites resolve to backend
- [ ] WebSocket `/ws/*` connects

---

## 7. Known Risks & Ongoing Issues

| Risk | Status |
|------|--------|
| ImportError on ModelManager | 🔧 Fixed — lazy import in predict/shap/quality |
| MWD_CORS_ORIGINS parse error | Unknown — not observed in new deploy |
| Missing GitHub integration → manual deploys only | User action required |
| NEXT_PUBLIC_API_URL not set on mwd-frontend | User action required |
| Python 3.12 in Railway container (base shows `python:3.10-slim`) | Unresolved — may cause pydantic-settings v2 differences |
| HEALTHCHECK directive on root Dockerfile removed | Intentionally excluded |

---

## 8. Next Actions for User

1. **Open Railway Dashboard** → `AI-MWD-Copilot` → `mwd-frontend` → Variables
   → Add `NEXT_PUBLIC_API_URL=https://mwd-backend-production.up.railway.app`

2. **After env var is set**, redeploy both services:
   ```bash
   railway up --service mwd-backend --detach
   railway up --service mwd-frontend --detach
   ```

3. **Verify** `GET https://mwd-backend-production.up.railway.app/health` → `{"status":"healthy"}`

---

## 9. Rollback Plan

If either service still fails after the Python fix:

1. **Backend rollback** — Revert `predict.py`, `shap.py`, `quality.py` to use direct `model_manager` imports, re-deploy.
2. **Frontend rollback** — Re-run `railway up --service mwd-frontend --detach` after adding env var.
3. **Domain rollback** — Railway domain persists across config changes; no DNS propagation needed.
