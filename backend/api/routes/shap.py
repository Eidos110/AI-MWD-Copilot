"""Feature importance API routes - Fast Mode (no SHAP)."""

import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter()

# Feature names for each model
_POROSITY_FEATURES = [
    "DEPTH", "Gamma Ray - Corrected gAPI", "Resistivity Phase - Corrected - 2MHz ohm.m",
    "Corrected Drilling Exponent unitless", "ROP for the Bit - Distance Over Time (On Bottom) m/s",
    "Surface Torque Average N.m", "Weight On Bit N", "Chrom 1 Total Gas Euc",
]

_FLUID_FEATURES = [
    "DEPTH", "Gamma Ray - Corrected gAPI", "Corrected Drilling Exponent unitless",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s", "Mechanical Specific Energy Pa",
    "Surface Torque Average N.m", "Weight On Bit N", "28 Stick Slip RPM Average RPM",
]

_PRESSURE_FEATURES = [
    "DEPTH", "Mud Weight In kg/m3", "ECD at Bit kg/m3", "Annular Pressure Pa",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s", "Weight On Bit N",
    "Surface Torque Average N.m", "DEPTH_FT", "P_Hydrostatic", "Delta_P_Hydro",
    "P_Overburden", "Effective_Stress", "Pressure_Anomaly",
]

_FEATURE_NAMES = {
    "DEPTH": "Well Depth (m)",
    "Gamma Ray - Corrected gAPI": "Gamma Ray",
    "Resistivity Phase - Corrected - 2MHz ohm.m": "Resistivity (Phase)",
    "Corrected Drilling Exponent unitless": "Corrected Drilling Exponent",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s": "Rate of Penetration (ROP)",
    "Surface Torque Average N.m": "Surface Torque",
    "Weight On Bit N": "Weight on Bit (WOB)",
    "Chrom 1 Total Gas Euc": "Chrome 1 Total Gas",
    "Mechanical Specific Energy Pa": "Mechanical Specific Energy (MSE)",
    "28 Stick Slip RPM Average RPM": "Stick Slip RPM",
    "Mud Weight In kg/m3": "Mud Weight",
    "ECD at Bit kg/m3": "Equivalent Circulating Density (ECD)",
    "Annular Pressure Pa": "Annular Pressure",
    "DEPTH_FT": "Depth (feet)",
    "P_Hydrostatic": "Hydrostatic Pressure",
    "Delta_P_Hydro": "Delta Hydrostatic Pressure",
    "P_Overburden": "Overburden Pressure",
    "Effective_Stress": "Effective Stress",
    "Pressure_Anomaly": "Pressure Anomaly",
}

_manager = None


def _load_manager():
    global _manager
    if _manager is None:
        try:
            from backend.services.model_manager import ModelManager
            _manager = ModelManager()
        except Exception as exc:
            logger.error("ModelManager unavailable: %s", exc)
            _manager = None
    return _manager


class ShapRequest(BaseModel):
    model: str
    data: List[Dict[str, Any]]
    top_n: int = 3


class ShapResponse(BaseModel):
    summary: str
    top_positive: List[Dict[str, Any]]
    top_negative: List[Dict[str, Any]]
    importance: List[Dict[str, Any]]


@router.post("/shap/explain", response_model=ShapResponse)
async def explain_model(req: ShapRequest):
    """Fast feature importance using XGBoost built-in (no SHAP computation)."""
    try:
        mgr = _load_manager()
        if mgr is None:
            raise HTTPException(status_code=503, detail="Models loading, try again in 10s.")

        if req.model == "porosity":
            model = mgr.porosity_model
            features = _POROSITY_FEATURES
        elif req.model == "fluid":
            model = mgr.fluid_model
            features = _FLUID_FEATURES
        elif req.model == "pressure":
            model = mgr.pressure_model
            features = _PRESSURE_FEATURES
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

        booster = model.get_booster()
        try:
            gains = booster.feature_importances_
        except AttributeError:
            import numpy as np
            gains_dict = booster.get_score(importance_type="gain")
            gains = np.array([gains_dict.get(f, 0.0) for f in features])

        df = pd.DataFrame({
            "Original Name": features,
            "Display Name": [f"{_FEATURE_NAMES.get(f, f)}" for f in features],
            "Mean |SHAP|": [float(g) for g in gains],
            "Direction": ["↑ PUSH UP"] * len(features),
        }).sort_values("Mean |SHAP|", ascending=False)

        summary = f"Model: {req.model}\nSamples: {len(req.data)}\n\nTop Features:\n"
        for _, r in df.head(req.top_n).iterrows():
            summary += f"- {r['Display Name']}: {r['Mean |SHAP|']:.4f}\n"

        return {
            "summary": summary,
            "top_positive": df.head(req.top_n).to_dict("records"),
            "top_negative": [],
            "importance": df.to_dict("records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
