# Railway Deployment Plan — AI-MWD-Copilot

> **Project**: AI-MWD-Copilot
> **Project Root**: `E:\Code\Well-Logging-AI-AWD-Copilot-Deepseek`
> **Target**: Railway (app.railway.com)
> **Akun**: GitHub yang sama dengan Railw

---

## 1. Overview

Deploy aplikasi ke Railway:
- **Backend**: FastAPI Python → Railway Service
- **Frontend**: Next.js → Railway Service

---

## 2. Prerequisites

### 2.1 Install Railway CLI (Windows)

**Via npm (recommended):**
```powershell
npm install -g @railway/cli
```

**Atau via direct download:**
1. Buka https://github.com/railwayapp/cli/releases
2. Download versi terbaru untuk Windows
3. Extract dan masukkan ke PATH

### 2.2 Verify Installation

```powershell
railway --version
```

### 2.3 Login

```powershell
railway login
# Buka browser untuk auth
```

---

## 3. Opsi Penerapan

### Opsi 1: Docker (Direct dari komputer ini)

**Kelebihan:**
- Build lokal - lebih cepat iteration
- Bisa test sebelum push
- Python version terkontrol di Dockerfile

**Langkah:**

1. **Login Railway:**
```powershell
railway login
railway init
# Pilih project baru atau existing
```

2. **Build Docker lokal:**
```powershell
docker build -t mwd-backend -f backend/Dockerfile .
docker build -t mwd-frontend -f frontend/Dockerfile .
```

3. **Push ke Railway:**
```powershell
railway up --service mwd-backend
# Atau via docker login
docker login containers.railway.app
railway connect
```

**Atau langsung provision tanpa docker:**
```powershell
railway up --dockerfile backend/Dockerfile
```

---

### Opsi 2: Tanpa Docker (Direct dari komputer ini)

**Kelebihan:**
- Lebih simple
- Railway auto-detect stack

**Langkah:**

1. **Login Railway:**
```powershell
railway login
railway init
```

2. **Deploy Backend:**
```powershell
railway up --service mwd-backend
# Set environment:
railway variables set MWD_CORS_ORIGINS='["*"]'
railway variables set PYTHONPATH=/app
```

3. **Deploy Frontend:**
```powershell
railway up --service mwd-frontend --rootDirectory frontend
railway variables set NEXT_PUBLIC_API_URL=https://mwd-backend.railway.app
```

---

### Opsi 3: Via GitHub (Pipeline Otomatis)

**Kelebihan:**
- Auto deploy setiap push
- No local CLI needed

**Konfigurasi:**

1. **Connect Repo di Railway Dashboard:**
   - Buka https://railway.app/new
   - Connect GitHub repo
   
2. **Setup Services:**

**Backend (tanpa Docker):**
| Field | Value |
|-------|-------|
| Service Name | `mwd-backend` |
| Root Directory | `.` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python runist.py` |

**Frontend:**
| Field | Value |
|-------|-------|
| Service Name | `mwd-frontend` |
| Root Directory | `frontend` |
| Build Command | `npm run build` |
| Start Command | `npm run start` |

3. **Set Environment Variables:**
   - Backend: `MWD_CORS_ORIGINS`, `PYTHONPATH`
   - Frontend: `NEXT_PUBLIC_API_URL`

---

## 4. Files yang Dibutuhkan

### Untuk Docker (Opsi 1/3):

**`backend/Dockerfile`** (sudah ada):
```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY backend/ /app/backend/
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MWD_CORS_ORIGINS=["*"]
EXPOSE 8000
CMD ["python", "-c", "import os; import uvicorn; uvicorn.run('backend.app:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))"]
```

**`frontend/Dockerfile`** (sudah ada):
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./frontend/
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Untuk Tanpa Docker (Opsi 2/3):

Pastikan files ini ada di root:
- `requirements.txt` ✓ (sudah di-commit)
- `runtime.txt` ✓ (python-3.10.13)

---

## 5. Environment Variables

### Backend
| Variable | Value |
|----------|-------|
| `MWD_CORS_ORIGINS` | `["*"]` |
| `PYTHONPATH` | `/app` |

### Frontend  
| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | (URL backend railway, contoh: `https://mwd-backend.railway.app`) |
| `NODE_ENV` | `production` |

---

## 6. Deployment Commands (Ringkas)

### Opsi 1 (Docker lokal):
```bash
# Login
railway login

# Deploy backend
railway up --service mwd-backend --dockerfile backend/Dockerfile

# Deploy frontend  
railway up --service mwd-frontend --dockerfile frontend/Dockerfile
```

### Opsi 2 (Tanpa Docker lokal):
```bash
railway login
railway up --service mwd-backend
railway up --service mwd-frontend --rootDirectory frontend
```

### Opsi 3 (GitHub):
```bash
# Setup di dashboard aja - auto deploy via git push
```

---

## 7. Troubleshooting

**Masalah:** Backend timeout saat startup
**Solusi:** Tambah health check atau start command benar

**Masalah:** Module not found
**Solusi:** PYTHONPATH=/app sudah diset

**Masalah:** 503 Service Unavailable
**Solusi:** Tunggu warmup (Cold start di free tier)

---

## 8. Verify Deployment

```bash
# Check backend
curl https://mwd-backend.railway.app/health

# Check frontend
curl https://mwd-frontend.railway.app
```

---

## 9. Catatan Penting

- Railway free tier: 500 jam/month (semua service collectively)
- Sleep после 5 hari inactive (but bisa disable)
- Custom domain bisa added di dashboard
- Logs bisa dicek di Railway dashboard atau CLI:
```bash
railway logs --service mwd-backend
```

## 10. Referensi

- Railway CLI Docs: https://docs.railway.app/reference/cli
- Railway Pricing: https://railway.app/pricing
- Deploy Button: https://railway.app/badge