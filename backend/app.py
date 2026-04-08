"""FastAPI application entry point for AI-Powered MWD Copilot."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from backend.core.config import settings
from backend.core.logging import setup_logging

logger = setup_logging()

# Global model manager instance
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global model_manager
    logger.info("Loading ML models...")
    from backend.services.model_manager import ModelManager

    model_manager = ModelManager()
    logger.info("Models loaded. Application ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-time ML system for drilling decision support",
    lifespan=lifespan,
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
from backend.api.routes import predict, data, quality, shap, health, interpret

app.include_router(health.router, tags=["health"])
app.include_router(predict.router, prefix="/api/v1", tags=["predict"])
app.include_router(data.router, prefix="/api/v1", tags=["data"])
app.include_router(quality.router, prefix="/api/v1", tags=["quality"])
app.include_router(shap.router, prefix="/api/v1", tags=["shap"])
app.include_router(interpret.router, prefix="/api/v1", tags=["interpret"])

# WebSocket
from backend.websocket.manager import router as ws_router

app.include_router(ws_router, tags=["websocket"])


def get_model_manager():
    """Dependency to get the global model manager."""
    return model_manager
