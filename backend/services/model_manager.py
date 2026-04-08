"""Model manager for loading and running predictions.

Manages loading, validation, and inference for three predictive models:
- Porosity Model: Predicts PHI_COMBINED (composite porosity).
- Fluid Model: Classifies fluid type (Background/Pay Zone/Potential Reservoir).
- Pressure Model: Estimates pore pressure (PSI).
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from backend.core.config import settings

logger = logging.getLogger(__name__)

MODEL_POROSITY = settings.MODEL_DIR / settings.MODEL_POROSITY
MODEL_FLUID = settings.MODEL_DIR / settings.MODEL_FLUID
MODEL_PRESSURE = settings.MODEL_DIR / settings.MODEL_PRESSURE
ENCODER_FLUID = settings.MODEL_DIR / settings.ENCODER_FLUID

FEATURES_POROSITY = [
    "DEPTH",
    "Gamma Ray - Corrected gAPI",
    "Resistivity Phase - Corrected - 2MHz ohm.m",
    "Corrected Drilling Exponent unitless",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s",
    "Surface Torque Average N.m",
    "Weight On Bit N",
    "Chrom 1 Total Gas Euc",
]

FEATURES_FLUID = [
    "DEPTH",
    "Gamma Ray - Corrected gAPI",
    "Corrected Drilling Exponent unitless",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s",
    "Mechanical Specific Energy Pa",
    "Surface Torque Average N.m",
    "Weight On Bit N",
    "28 Stick Slip RPM Average RPM",
]

FEATURES_PRESSURE = [
    "DEPTH",
    "Mud Weight In kg/m3",
    "ECD at Bit kg/m3",
    "Annular Pressure Pa",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s",
    "Weight On Bit N",
    "Surface Torque Average N.m",
    "DEPTH_FT",
    "P_Hydrostatic",
    "Delta_P_Hydro",
    "P_Overburden",
    "Effective_Stress",
    "Pressure_Anomaly",
]

MINIMAL_FEATURES = [
    "DEPTH",
    "Weight On Bit N",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s",
]


class ModelManager:
    """Manages loading, validation, and inference for three predictive models."""

    def __init__(self):
        """Load all three models and the fluid label encoder from disk."""
        try:
            self.porosity_model = joblib.load(str(MODEL_POROSITY))
            self.fluid_model = joblib.load(str(MODEL_FLUID))
            self.pressure_model = joblib.load(str(MODEL_PRESSURE))
            self.fluid_encoder = joblib.load(str(ENCODER_FLUID))
            logger.info("All models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def predict_porosity(self, df: pd.DataFrame) -> np.ndarray:
        """Predict porosity values (PHI_COMBINED)."""
        X = self._safe_select(df, FEATURES_POROSITY, "porosity")
        if X.empty:
            raise ValueError("No valid features available for porosity prediction")
        X = X.fillna(X.mean())
        return self.porosity_model.predict(X)

    def predict_fluid(self, df: pd.DataFrame) -> tuple:
        """Predict fluid type and class probabilities."""
        X = self._safe_select(df, FEATURES_FLUID, "fluid")
        if X.empty:
            raise ValueError("No valid features available for fluid prediction")
        X = X.fillna(X.mean())
        pred_class = self.fluid_model.predict(X)
        pred_proba = self.fluid_model.predict_proba(X)
        return self.fluid_encoder.inverse_transform(pred_class), pred_proba

    def predict_pressure(self, df: pd.DataFrame) -> np.ndarray | None:
        """Predict pore pressure using the full FEATURES_PRESSURE list."""
        try:
            X = self._safe_select(df, FEATURES_PRESSURE, "pressure", force_full=True)
            if X.empty:
                raise ValueError("No valid features available for pressure prediction")

            X = X.fillna(X.mean()).fillna(0)

            expected_features = self.pressure_model.get_booster().feature_names

            missing_for_model = [f for f in expected_features if f not in X.columns]
            if missing_for_model:
                logger.warning(
                    f"Pressure model expected extra features; adding missing: {missing_for_model}"
                )
                for f in missing_for_model:
                    X[f] = 0

            X = X[expected_features]
            return self.pressure_model.predict(X)
        except Exception as e:
            logger.warning(f"Pressure prediction unavailable: {e}")
            return None

    def _safe_select(
        self,
        df: pd.DataFrame,
        features: list,
        model_name: str,
        force_full: bool = False,
    ) -> pd.DataFrame:
        """Select features with fallback and error handling."""
        if force_full:
            selected = df.reindex(columns=features).copy()
            return selected

        available = [f for f in features if f in df.columns]

        if len(available) < 3:
            logger.warning(
                f"Not enough features for {model_name} model. Using minimal set."
            )
            available = [f for f in MINIMAL_FEATURES if f in df.columns]

        if len(available) == 0:
            logger.error(f"No valid features found for {model_name} prediction.")
            raise ValueError(f"Cannot proceed: no valid features for {model_name}")

        selected = df[available].copy()

        if selected.isna().all().any():
            problematic = selected.columns[selected.isna().all()].tolist()
            logger.warning(
                f"Columns with all NaN values in {model_name}: {problematic}"
            )

        return selected
