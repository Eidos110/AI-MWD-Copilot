# AI-Powered MWD Copilot

Real-Time Machine Learning for Drilling Decision Support

**Prepared by**: Eidos/W_Isnal, Data Science & Geophysics

**Version**: 3.0 (FastAPI + Next.js)  
**Date**: January 2026


## Architecture Overview

This is a modern full-stack application with:

- **Backend**: FastAPI (Python 3.10+) - ML inference, data processing
- **Frontend**: Next.js (React) - Interactive dashboard
- **Deployment**: Docker Compose for production, manual dev mode supported

### Core Features

| Feature | Description |
|---------|-------------|
| **Porosity Prediction** | XGBoost regressor predicting PHI_COMBINED |
| **Fluid Classification** | 3-class classifier (Potential Reservoir, Pay Zone, Background) |
| **Pore Pressure Estimation** | XGBoost regressor with Rehm & McClendon method |
| **SHAP Explainability** | TreeExplainer for all three models |
| **Data Quality Assessment** | Missing values, outliers, sensor health scoring |
| **Real-time Streaming** | WebSocket support for live predictions |

---

## Quick Start (Docker Compose)

### Prerequisites

- Docker 24.0+
- Docker Compose v2.0+

### Run the Application

```bash
docker-compose up --build
```

The application will be available at:

| Service | Local URL | Railway URL |
|---------|-----------|-------------|
| Frontend (Next.js) | http://localhost:3000 | _(pending — mwd-frontend not yet deployed)_ |
| Backend (FastAPI) | http://localhost:8000 | https://mwd-backend-production.up.railway.app |
| API Docs | http://localhost:8000/docs | https://mwd-backend-production.up.railway.app/docs |

### Stop the Application

```bash
docker-compose down
```

---

## Manual Development Mode

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/api/v1/health` | API version info |

### Prediction Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict/porosity` | Predict porosity values |
| POST | `/api/v1/predict/fluid` | Predict fluid type classification |
| POST | `/api/v1/predict/pressure` | Predict pore pressure |
| POST | `/api/v1/predict/all` | Run all three predictions |

### Data Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/data/sample` | Get default dataset |
| POST | `/api/v1/data/upload` | Upload CSV/Excel file |
| POST | `/api/v1/data/validate` | Validate uploaded file |
| GET | `/api/v1/data/columns` | Get available columns |

### Quality & Explainability

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/quality/report` | Generate data quality report |
| POST | `/api/v1/shap/explain` | Generate SHAP explanations |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws/stream` | Real-time streaming for predictions |

---

## API Request Examples

### Predict All Models

```bash
curl -X POST http://localhost:8000/api/v1/predict/all \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"DEPTH": 2500, "GR": 50, "RD": 200, "ROP": 10, "WOB": 50000},
      {"DEPTH": 2510, "GR": 55, "RD": 180, "ROP": 12, "WOB": 52000}
    ],
    "include_confidence": true
  }'
```

### Upload Data

```bash
curl -X POST http://localhost:8000/api/v1/data/upload \
  -F "file=@data.csv"
```

### Get Quality Report

```bash
curl -X POST http://localhost:8000/api/v1/quality/report \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"DEPTH": 2500, "GR": 50, "RD": 200, "ROP": 10}
    ]
  }'
```

---

## Project Structure

```
well-logging-ai-awd-copilot-deepseek/
├── backend/
│   ├── app.py                 # FastAPI entry point
│   ├── api/routes/            # API endpoints
│   │   ├── predict.py        # Prediction endpoints
│   │   ├── data.py           # Data upload/retrieval
│   │   ├── quality.py        # Quality assessment
│   │   ├── shap.py           # SHAP explainability
│   │   └── health.py         # Health checks
│   ├── services/             # Business logic
│   │   ├── model_manager.py  # ML model management
│   │   ├── data_loader.py   # Data loading
│   │   ├── predictions.py   # Confidence & intervals
│   │   ├── shap_explainer.py
│   │   ├── data_quality.py
│   │   └── targets.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── websocket/
│   │   └── manager.py       # WebSocket streaming
│   ├── models/               # Trained XGBoost models
│   ├── data/                 # Sample datasets
│   └── requirements.txt
│
├── frontend/
│   ├── app/              # Next.js app router (pages)
│   ├── components/       # React components
│   ├── lib/              # Utilities & helpers
│   ├── stores/           # State management
│   ├── types/            # TypeScript types
│   ├── public/           # Static assets
│   ├── package.json
│   └── Dockerfile        # Multi-stage build (Node 20, standalone output)
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

## Model Information

### Input Features

| Model | Features |
|-------|----------|
| **Porosity** | GR, Resistivity, ROP, WOB, Gas |
| **Fluid** | GR, MSE, ROP, Stick-Slip, Torque |
| **Pressure** | MW, ECD, ROP, WOB, DTC, EXP |

### Output Variables

| Variable | Description | Unit |
|----------|-------------|------|
| PHI_COMBINED | Predicted porosity | fraction |
| FLUID_CLASS | Fluid type classification | categorical |
| PREDICTED_PORE_PRESSURE_PSI | Pore pressure | psi |

---

## Docker Configuration

### Local — Docker Compose

The `docker-compose.yml` sets up:

- **backend**: Port 8000, loads models from `./backend/models`
- **frontend**: Port 3000, proxies API to backend
- **Network**: Bridge network for inter-service communication

### Production — Railway Deployment

The backend service uses the **root `Dockerfile`** (multi-purpose image built from `python:3.10-slim`). The Dockerfile:

1. Installs Python deps from `backend/requirements.txt`
2. Copies `backend/` source and `start.sh`
3. `start.sh` reads Railway's injected `$PORT`, defaulting to `8000`, and runs `exec uvicorn backend.app:app --host 0.0.0.0 --port "$PORT"`
4. ML models are loaded in a **background daemon thread** at import time so the `/health` endpoint is responsive immediately

```dockerfile
# Railway backend service uses repo-root Dockerfile
CMD ["/app/start.sh"]   # reads $PORT from Railway, defaults to 8000
```

The frontend service uses a **multi-stage `frontend/Dockerfile`** based on Node 20:

1. Builder stage: `npm ci && npm run build` → output to `.next/standalone`
2. Runtime stage: copies `.next/standalone`, `.next/static`, `public/`
3. Starts via `node server.js` (generated by Next.js build)

#### Health Checks

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Backend health — returns `{"status":"healthy"}` |
| `GET /` | Backend root — returns `{"status":"healthy","version":"3.0.0"}` |
| `GET /health` (frontend) | Proxied to backend via Next.js rewrites |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MWD_CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `["*"]` |
| `MWD_LOG_LEVEL` | Logging level | `info` |
| `NEXT_PUBLIC_API_URL` | API base URL for the frontend | `http://localhost:8000` |


---

---

## Railway Deployment

> **Railway Project**: AI-MWD-Copilot  
> **Environment**: production  
> **Backend URL**: https://mwd-backend-production.up.railway.app  
> **Frontend URL**: _(pending — service `mwd-frontend` not yet live)_

### Services

| Service | ID | Status |
|---------|-----|--------|
| `mwd-backend` | `25d24257` | 🔄 Fresh deploy triggered |
| `mwd-frontend` | `5876f291` | ⏳ Config + env vars pending |

### Prerequisites

1. **Link GitHub to Railway** — required for `railway up` and auto-deploys  
   Railway dashboard → Settings → Integrations → GitHub

2. **Set environment variables** on Railway:
   - `MWD_CORS_ORIGINS` → `["*"]` (on backend)
   - `NEXT_PUBLIC_API_URL` → `https://mwd-backend-production.up.railway.app` (on frontend)

### Manual Deploy (via CLI)

```bash
# Backend (from repo root — uses root Dockerfile)
railway up --service mwd-backend --detach

# Frontend (uses frontend/Dockerfile multi-stage build)
railway up --service mwd-frontend --detach
```

### Architecture Notes

- Railway injects `$PORT` at runtime; `start.sh` reads it and defaults to `8000`
- Backend loads ML models in a daemon background thread — `/health` is available immediately
- Frontend rewrites `/api/*` → backend and `/ws/*` → backend WebSocket
- CORS is set to `["*"]` in both dev and production (tighten in production)

---

## Testing

```bash
# Backend tests
cd backend
pytest tests/ -v
```

---

## License

This project is proprietary R&D. Distribution restricted to authorized personnel.

---

**Last Updated**: March 2026
