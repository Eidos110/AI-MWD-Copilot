"""Pydantic-based application settings.

Replaces config.yaml with environment-variable-driven configuration.
All settings can be overridden via environment variables with the prefix MWD_.
"""

import os
import json
import logging
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

logger = logging.getLogger(__name__)

_CORS_ENV_KEY = "MWD_CORS_ORIGINS"
_CORS_ORIGINS_DEFAULT = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]


def _parse_cors_origins(raw: str | None) -> List[str]:
    """Parse CORS_ORIGINS from MWD_CORS_ORIGINS env var (JSON array or CSV).

    Tries JSON first (handles Rail`[' *\"]`), then falls back to comma-separated.
    Returns the hardcoded default when the env var is absent or unparseable.
    """
    if not raw:
        return _CORS_ORIGINS_DEFAULT[:]
    raw = raw.strip()
    if not raw:
        return _CORS_ORIGINS_DEFAULT[:]
    # Try JSON array: '["*"]', '["http://a", "http://b"]'
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(x) for x in val]
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to comma-separated raw string
    return [s.strip() for s in raw.split(",") if s.strip()]


class Settings(BaseSettings):
    """Application settings with MWD_ env-var prefix.

    CORS_ORIGINS needs special handling because pydantic-settings v2 tries to
    JSON-decode every complex-type field read from ENV and raises SettingsError
    when the raw string is not valid JSON array syntax. We therefore set
    CORS_ORIGINS via the explicit constructor kwarg (``Settings(CORS_ORIGINS=…)``)
    in the factory below, completely bypassing the ENV-source reader.
    """

    APP_NAME: str = "AI-Powered MWD Copilot"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False

    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_DIR: Path = Path(__file__).parent.parent / "models"
    DATA_DIR: Path = Path(__file__).parent.parent / "data"
    DATA_PATH: Path = Path(__file__).parent.parent / "data" / "ready_modelling.csv"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    CORS_ORIGINS: List[str] = _CORS_ORIGINS_DEFAULT[:]

    LOG_LEVEL: str = "INFO"

    DEFAULT_MIN_DEPTH: int = 2000
    DEFAULT_MAX_DEPTH: int = 2500
    MAX_UPLOAD_SIZE_MB: int = 200

    MODEL_POROSITY: str = "xgb_phi_model.pkl"
    MODEL_FLUID: str = "xgb_fluid_model.pkl"
    MODEL_PRESSURE: str = "xgb_pp_model_feat.pkl"
    ENCODER_FLUID: str = "le.pkl"

    class Config:
        env_prefix = "MWD_"
        case_sensitive = False


def _init_settings() -> Settings:
    """Create the global settings singleton.

    Reads MWD_CORS_ORIGINS from the environment with a manual parser before
    constructing *Settings*, passing the result as a keyword argument so
    pydantic-settings never touches — and never chokes on — that field's ENV value.
    """
    cors_override = _parse_cors_origins(os.environ.get(_CORS_ENV_KEY))
    return Settings(CORS_ORIGINS=cors_override)   # type: ignore[call-arg]


try:
    settings = _init_settings()
    logger.info("Settings loaded: CORS_ORIGINS=%s", settings.CORS_ORIGINS)
except Exception as exc:
    logger.error("Settings FAILED (%s): %s\n%s", type(exc).__name__, exc,
                 __import__("traceback").format_exc())

    class _FallbackSettings:
        """Minimal settings used when Settings() cannot be constructed."""
        APP_NAME = "AI-Powered MWD Copilot"
        APP_VERSION = "3.0.0"
        DEBUG = False
        BASE_DIR = Path(__file__).parent.parent
        MODEL_DIR = Path(__file__).parent.parent / "models"
        DATA_DIR = Path(__file__).parent.parent / "data"
        DATA_PATH = Path(__file__).parent.parent / "data" / "ready_modelling.csv"
        HOST = "0.0.0.0"
        PORT = 8000
        CORS_ORIGINS = _CORS_ORIGINS_DEFAULT[:]
        LOG_LEVEL = "INFO"
        DEFAULT_MIN_DEPTH = 2000
        DEFAULT_MAX_DEPTH = 2500
        MAX_UPLOAD_SIZE_MB = 200
        MODEL_POROSITY = "xgb_phi_model.pkl"
        MODEL_FLUID = "xgb_fluid_model.pkl"
        MODEL_PRESSURE = "xgb_pp_model_feat.pkl"
        ENCODER_FLUID = "le.pkl"

    settings = _FallbackSettings()  # type: ignore[assignment]
