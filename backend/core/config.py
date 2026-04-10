"""Pydantic-based application settings.

Replaces config.yaml with environment-variable-driven configuration.
All settings can be overridden via environment variables with the prefix MWD_.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI-Powered MWD Copilot"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_DIR: Path = Path(__file__).parent.parent / "models"
    DATA_DIR: Path = Path(__file__).parent.parent / "data"
    DATA_PATH: Path = Path(__file__).parent.parent / "data" / "ready_modelling.csv"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]

    # Logging
    LOG_LEVEL: str = "INFO"

    # Data
    DEFAULT_MIN_DEPTH: int = 2000
    DEFAULT_MAX_DEPTH: int = 2500
    MAX_UPLOAD_SIZE_MB: int = 200

    # Models
    MODEL_POROSITY: str = "xgb_phi_model.pkl"
    MODEL_FLUID: str = "xgb_fluid_model.pkl"
    MODEL_PRESSURE: str = "xgb_pp_model_feat.pkl"
    ENCODER_FLUID: str = "le.pkl"

    class Config:
        env_prefix = "MWD_"
        case_sensitive = False


settings = Settings()
