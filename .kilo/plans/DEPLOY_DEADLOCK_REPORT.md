# Railway Deployment - Deadlock Report
  
## Status Terkini
- Backend (mwd-backend): ✅ Healthy on https://mwd-backend-production-f958.up.railway.app
- Frontend (mwd-frontend): ❌ FAILED - "Cannot redeploy without a snapshot"
- Railway CLI: ⚠️ Timeout pada semua request ke backboard.railway.com
  
## Yang Sudah Dilakukan (Commit di Git)
1. `frontend/Dockerfile` - dipulihkan dengan path COPY yang benar
2. `railway.json` - dikonfigurasi dengan:
   - mwd-backend (root: backend, build: pip install, start: uvicorn)
   - mwd-frontend (root: frontend, dockerfilePath: Dockerfile)
3. `MWD_CORS_ORIGINS` di-set ke `["*"]` (valid JSON) via Railway CLI
  
## Kendala
- Snapshot lama (build sebelumnya) tidak mengandung Dockerfile yang valid
- CLI tidak dapat trigger build baru karena timeout API
- Service `mwd-frontend` belum terhubung ke GitHub webhook untuk auto-deploy
  
## Tindakan Manual Diperlukan (Railway Dashboard)
1. Buka https://railway.com/project/f899cfb8-dd83-4c81-aae8-8ee931b0cc5b
2. Pilih service **mwd-frontend**
3. Di **Settings → Build & Deploy**:
   - Builder: pilih **DOCKERFILE**
   - Root Directory: `frontend`
   - Dockerfile Path: `Dockerfile`
4. Di **Variables**, tambah:
   - `NEXT_PUBLIC_API_URL` = `https://mwd-backend-production-f958.up.railway.app`
   - `NODE_ENV` = `production`
5. Klik **Deploy** (bukan Redeploy) untuk membuat snapshot baru dari commit terbaru
6. Tunggu build selesai, lalu akses https://mwd-frontend-production.up.railway.app
  
## Alternatif Jika Dashboard Tidak Merespons
- Buat service baru di Railway Dashboard dengan nama `mwd-frontend-v2`
- Hubungkan ke GitHub repo `Eidos110/AI-MWD-Copilot`
- Branch `main` akan auto-deploy