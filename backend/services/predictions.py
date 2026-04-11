"""Prediction confidence and uncertainty quantification module.

Provides utilities to:
- Estimate prediction confidence from model internals
- Add uncertainty bands to predictions
- Export predictions with confidence scores (in-memory, no file I/O)
"""

import numpy as np

# Fix for deprecated np.int in newer numpy versions
if not hasattr(np, "int"):
    np.int = int

import pandas as pd
from typing import Tuple, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def compute_prediction_confidence(
    model, X_test: pd.DataFrame, y_pred: np.ndarray
) -> np.ndarray:
    """Estimate prediction confidence from XGBoost model internals.

    Uses prediction variance across different tree iteration depths.
    Returns confidence scores (0-1 scale).
    """
    try:
        if hasattr(model, "predict") and hasattr(model, "get_booster"):
            booster = model.get_booster()
            n_trees = getattr(booster, "best_ntree_limit", None)
            if n_trees is None:
                n_trees = booster.num_boosted_rounds()

            predictions_all = []
            for ntrees in [
                max(1, n_trees // 4),
                max(1, n_trees // 2),
                max(1, 3 * n_trees // 4),
                n_trees,
            ]:
                try:
                    pred = model.predict(X_test, iteration_range=(0, ntrees))
                    predictions_all.append(pred)
                except Exception:
                    break

            if len(predictions_all) > 1:
                predictions_all = np.vstack(predictions_all)
                variance = np.var(predictions_all, axis=0)
                mean_pred = np.mean(predictions_all, axis=0)

                with np.errstate(divide="ignore", invalid="ignore"):
                    cv = np.sqrt(variance) / (np.abs(mean_pred) + 1e-6)
                    confidence = 1.0 / (1.0 + cv)
                    confidence = np.clip(confidence, 0, 1)
                    return confidence

        return np.ones(len(y_pred))

    except Exception:
        return np.ones(len(y_pred))


def compute_prediction_intervals(
    y_pred: np.ndarray, confidence: np.ndarray, ci: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate prediction intervals based on confidence scores."""
    z_score = 1.96 if ci == 0.95 else 1.645 if ci == 0.90 else 1.0
    margin = z_score * (1 - confidence) * np.abs(y_pred + 1e-6)
    lower = y_pred - margin
    upper = y_pred + margin
    return lower, upper


def export_predictions_csv(df: pd.DataFrame, predictions: dict) -> str:
    """Export predictions with confidence to CSV string (in-memory)."""
    export_df = (
        df[["DEPTH"]].copy()
        if "DEPTH" in df.columns
        else pd.DataFrame({"Row": range(len(df))})
    )
    for key, values in predictions.items():
        if isinstance(values, np.ndarray):
            export_df[key] = values
    return export_df.to_csv(index=False)


def export_predictions_json(
    df: pd.DataFrame, predictions: dict, metadata: dict = None
) -> str:
    """Export predictions with metadata to JSON string (in-memory)."""
    export_data = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "num_samples": len(df),
            "num_features": len(df.columns),
            **(metadata or {}),
        },
        "predictions": {},
    }
    if "DEPTH" in df.columns:
        export_data["predictions"]["DEPTH"] = df["DEPTH"].tolist()
    for key, values in predictions.items():
        if isinstance(values, np.ndarray):
            export_data["predictions"][key] = values.tolist()
    return json.dumps(export_data, indent=2)


def create_prediction_report(
    df: pd.DataFrame, predictions: dict, confidences: dict = None
) -> pd.DataFrame:
    """Create a comprehensive prediction report DataFrame."""
    report = (
        df[["DEPTH"]].copy()
        if "DEPTH" in df.columns
        else pd.DataFrame(index=range(len(df)))
    )

    if "PHI_COMBINED" in df.columns:
        report["Porosity (True)"] = df["PHI_COMBINED"]
    if "phi_pred" in predictions:
        report["Porosity (Predicted)"] = predictions["phi_pred"]
        if confidences and "phi_conf" in confidences:
            report["Porosity Confidence"] = confidences["phi_conf"]
            lower, upper = compute_prediction_intervals(
                predictions["phi_pred"], confidences["phi_conf"]
            )
            report["Porosity Lower Bound"] = lower
            report["Porosity Upper Bound"] = upper

    if "FLUID_CLASS" in df.columns:
        report["Fluid Type (True)"] = df["FLUID_CLASS"]
    if "fluid_pred" in predictions:
        report["Fluid Type (Predicted)"] = predictions["fluid_pred"]

    if "PREDICTED_PORE_PRESSURE_PSI" in df.columns:
        report["Pore Pressure (True, PSI)"] = df["PREDICTED_PORE_PRESSURE_PSI"]
    if "pp_pred" in predictions:
        report["Pore Pressure (Predicted, PSI)"] = predictions["pp_pred"]
        if confidences and "pp_conf" in confidences:
            report["Pressure Confidence"] = confidences["pp_conf"]
            lower, upper = compute_prediction_intervals(
                predictions["pp_pred"], confidences["pp_conf"]
            )
            report["Pressure Lower Bound (PSI)"] = lower
            report["Pressure Upper Bound (PSI)"] = upper

    return report
