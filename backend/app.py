"""FastAPI application entry point for AI-Powered MWD Copilot."""

import logging
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

logger = logging.getLogger(__name__)

model_manager = None
_model_lock = threading.Lock()


def _load_models_sync():
    """Load ML models in a background thread (non-blocking for event loop)."""
    global model_manager
    try:
        logger.info("Loading ML models...")
        from backend.services.model_manager import ModelManager  # noqa: PLC0415

        with _model_lock:
            model_manager = ModelManager()
        logger.info("Models loaded. Application ready.")
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to load models: {e}")
        with _model_lock:
            model_manager = None
        logger.info("Starting without models.")


def _start_background_model_loading():
    """Spawn daemon thread to load models so /health is responsive immediately."""
    threading.Thread(target=_load_models_sync, daemon=True).start()


app = FastAPI(
    title="AI-Powered MWD Copilot",
    version="3.0.0",
    description="Real-time ML system for drilling decision support",
)

# Kick off model loading as soon as the module is imported (non-blocking).
_start_background_model_loading()

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"status": "healthy", "service": "mwd-copilot-api", "version": "3.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mwd-copilot-api"}


from backend.api.routes import (
    predict,
    data,
    quality,
    shap,
    health as health_route,
    interpret,
)

app.include_router(health_route.router, tags=["health"])
app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
app.include_router(data.router, prefix="/api/v1", tags=["data"])
app.include_router(quality.router, prefix="/api/v1", tags=["quality"])
app.include_router(shap.router, prefix="/api/v1", tags=["shap"])
app.include_router(interpret.router, prefix="/api/v1", tags=["interpret"])

from backend.websocket.manager import router as ws_router

app.include_router(ws_router, tags=["websocket"])


def get_model_manager():
    """Dependency to get the global model manager."""
    global model_manager
    if model_manager is None:
        logger.warning("Model manager not initialized yet. Try again shortly.")
    return model_manager
