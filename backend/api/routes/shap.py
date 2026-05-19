"""SHAP model interpretability API routes — with safe lazy imports."""

import logging
import numpy as np

if not hasattr(np, "int"):
    np.int = int

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)
router = APIRouter()

# --- module-level feature sets (safe: no model_manager dependency) ----------
_POROSITY_FEATURES = [
    "DEPTH",
    "Gamma Ray - Corrected gAPI",
    "Resistivity Phase - Corrected - 2MHz ohm.m",
    "Corrected Drilling Exponent unitless",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s",
    "Surface Torque Average N.m",
    "Weight On Bit N",
    "Chrom 1 Total Gas Euc",
]

_FLUID_FEATURES = [
    "DEPTH",
    "Gamma Ray - Corrected gAPI",
    "Corrected Drilling Exponent unitless",
    "ROP for the Bit - Distance Over Time (On Bottom) m/s",
    "Mechanical Specific Energy Pa",
    "Surface Torque Average N.m",
    "Weight On Bit N",
    "28 Stick Slip RPM Average RPM",
]

_PRESSURE_FEATURES = [
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


# --- lazy ModelManager ------------------------------------------------------
_manager: Optional["ModelManager"] = None


def _load_manager():
    """Lazily import and instantiate the real ModelManager on first use."""
    global _manager
    if _manager is None:
        try:
            from backend.services.model_manager import ModelManager  # noqa: PLC0415
            _manager = ModelManager()
        except Exception as exc:
            logger.error("ModelManager unavailable: %s", exc)
            # Use a stub so SHAP still returns a helpful error rather than 500.
            class _Stub:
                porosity_model = None
                fluid_model = None
                pressure_model = None
                def _safe_select(self, *a, **kw):
                    raise RuntimeError("ModelManager unavailable — models not loaded yet.")
            _manager = _Stub()
    return _manager


# --- route handlers --------------------------------------------------------
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


def _get_shap_func():
    """Lazily import the SHAP interpreter."""
    from backend.services.shap_explainer import get_shap_interpretation  # noqa: PLC0415
    return get_shap_interpretation


@router.post("/shap/explain", response_model=ShapResponse)
async def explain_model(req: ShapRequest):
    """Generate SHAP explanation for a specific model."""
    try:
        mgr = _load_manager()
        df = pd.DataFrame(req.data)

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")

        # Select model and features
        if req.model == "porosity":
            model = mgr.porosity_model
            features = _POROSITY_FEATURES
            X = mgr._safe_select(df, features, "porosity")
        elif req.model == "fluid":
            model = mgr.fluid_model
            features = _FLUID_FEATURES
            X = mgr._safe_select(df, features, "fluid")
        elif req.model == "pressure":
            model = mgr.pressure_model
            X = mgr._safe_select(df, _PRESSURE_FEATURES, "pressure", force_full=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

        if X.empty:
            raise HTTPException(status_code=400, detail="Not enough valid features")

        X = X.fillna(X.mean()).fillna(0)
        sample = X.sample(n=min(req.max_samples, len(X)), random_state=42)

        get_shap_interpretation = _get_shap_func()
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error("SHAP explanation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
