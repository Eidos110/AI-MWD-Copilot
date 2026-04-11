"""SHAP model interpretability API routes."""

import logging
import numpy as np

# Fix for deprecated np.int in newer numpy versions
if not hasattr(np, "int"):
    np.int = int

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.services.model_manager import (
    ModelManager,
    FEATURES_POROSITY,
    FEATURES_FLUID,
    FEATURES_PRESSURE,
)
from backend.services.shap_explainer import get_shap_interpretation

logger = logging.getLogger(__name__)
router = APIRouter()

_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        from backend.app import get_model_manager

        _manager = get_model_manager()
    return _manager


SHAP_FEATURE_DISPLAY_NAMES = {
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


class ShapRequest(BaseModel):
    model: str  # "porosity", "fluid", "pressure"
    data: List[Dict[str, Any]]
    top_n: int = 5
    max_samples: int = 500


class ShapResponse(BaseModel):
    summary: str
    top_positive: List[Dict[str, Any]]
    top_negative: List[Dict[str, Any]]
    importance: List[Dict[str, Any]]


@router.post("/shap/explain", response_model=ShapResponse)
async def explain_model(req: ShapRequest):
    """Generate SHAP explanation for a specific model."""
    try:
        mgr = get_manager()
        df = pd.DataFrame(req.data)

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")

        # Select model and features
        if req.model == "porosity":
            model = mgr.porosity_model
            features = FEATURES_POROSITY
            X = mgr._safe_select(df, features, "porosity")
        elif req.model == "fluid":
            model = mgr.fluid_model
            features = FEATURES_FLUID
            X = mgr._safe_select(df, features, "fluid")
        elif req.model == "pressure":
            model = mgr.pressure_model
            features = FEATURES_PRESSURE
            X = mgr._safe_select(df, features, "pressure", force_full=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

        if X.empty:
            raise HTTPException(status_code=400, detail="Not enough valid features")

        X = X.fillna(X.mean()).fillna(0)
        sample = X.sample(n=min(req.max_samples, len(X)), random_state=42)

        interpretation = get_shap_interpretation(
            model,
            sample,
            feature_names=sample.columns.tolist(),
            display_names=SHAP_FEATURE_DISPLAY_NAMES,
            top_n=req.top_n,
        )

        importance_list = []
        if interpretation["data"] is not None:
            importance_list = interpretation["data"].to_dict(orient="records")

        return {
            "summary": interpretation["summary"],
            "top_positive": interpretation["top_positive"],
            "top_negative": interpretation["top_negative"],
            "importance": importance_list,
        }
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
