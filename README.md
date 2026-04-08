=======
# AI-Powered MWD Copilot

Real-Time Machine Learning for Drilling Decision Support

**Prepared by**: Eidos/W_Isnal, Data Science & Geophysics
**Version**: 3.0 (FastAPI + Next.js)  
**Date**: January 2026
---

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

| Service | URL |
|---------|-----|
| Frontend (Next.js) | http://localhost:3000 |
| Backend (FastAPI) | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

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
│   ├── src/
│   │   ├── app/             # Next.js app router
│   │   ├── components/      # React components
│   │   └── lib/             # Utilities
│   ├── package.json
│   └── Dockerfile
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

The `docker-compose.yml` sets up:

- **backend**: Port 8000, loads models from `./backend/models`
- **frontend**: Port 3000, proxies API to backend
- **Network**: Bridge network for inter-service communication

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| MWD_CORS_ORIGINS | Allowed CORS origins | ["http://localhost:3000"] |
| MWD_LOG_LEVEL | Logging level | info |
| NEXT_PUBLIC_API_URL | API base URL | http://localhost:8000 |

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
