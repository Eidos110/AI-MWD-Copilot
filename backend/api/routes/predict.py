"""Predict API routes — lazy imports so route module is safe to import
even if backend.services.model_manager is not yet available."""

import logging
import numpy as np

if not hasattr(np, "int"):
    np.int = int

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter()

# Global model manager reference — None until first use.
_manager: Optional["ModelManager"] = None


class _StubModelManager:
    """Fallback stub used when the real ModelManager cannot be imported.
    This keeps the import / module-level annotation from crashing the
    app on startup so FASTAPI always responds, even without cached models.
    """
    def predict_porosity(self, df):
        raise RuntimeError("ModelManager unavailable — models not loaded yet.")
    def predict_fluid(self, df):
        raise RuntimeError("ModelManager unavailable — models not loaded yet.")
    def predict_pressure(self, df):
        raise RuntimeError("ModelManager unavailable — models not loaded yet.")


def _load_manager():
    """Lazily import and instantiate the real ModelManager on first use."""
    global _manager
    if _manager is None:
        try:
            from backend.services.model_manager import ModelManager  # noqa: PLC0415
            _manager = ModelManager()
        except Exception as exc:
            logger.error("ModelManager unavailable: %s", exc)
            _manager = _StubModelManager()
    return _manager


def get_manager():
    """Return the (possibly stub) model manager."""
    return _load_manager()


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]
    depth_range: Optional[Dict[str, float]] = None
    include_confidence: bool = True


class PorosityResponse(BaseModel):
    predictions: List[float]
    confidence: Optional[List[float]] = None
    intervals: Optional[Dict[str, List[float]]] = None


class FluidResponse(BaseModel):
    predictions: List[str]
    probabilities: List[List[float]]
    classes: List[str]


class PressureResponse(BaseModel):
    predictions: List[float]
    confidence: Optional[List[float]] = None
    intervals: Optional[Dict[str, List[float]]] = None


class AllPredictionsResponse(BaseModel):
    porosity: Optional[PorosityResponse] = None
    fluid: Optional[FluidResponse] = None
    pressure: Optional[PressureResponse] = None
    report: Optional[List[Dict[str, Any]]] = None


def _df_from_request(req: PredictRequest) -> pd.DataFrame:
    """Convert request data to DataFrame, apply depth filter."""
    df = pd.DataFrame(req.data)
    if req.depth_range:
        mask = (df["DEPTH"] >= req.depth_range.get("min", -np.inf)) & (
            df["DEPTH"] <= req.depth_range.get("max", np.inf)
        )
        df = df[mask].copy()
    return df


@router.post("/predict/porosity", response_model=PorosityResponse)
async def predict_porosity(req: PredictRequest):
    """Predict porosity values."""
    try:
        mgr = get_manager()
        df = _df_from_request(req)
        preds = mgr.predict_porosity(df).tolist()
        return PorosityResponse(predictions=preds)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/fluid", response_model=FluidResponse)
async def predict_fluid(req: PredictRequest):
    """Predict fluid type and class probabilities."""
    try:
        mgr = get_manager()
        df = _df_from_request(req)
        preds, proba = mgr.predict_fluid(df)
        return FluidResponse(
            predictions=preds.tolist(),
            probabilities=proba.tolist(),
            classes=["Background", "Potential Reservoir", "Pay Zone"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/pressure", response_model=PressureResponse)
async def predict_pressure(req: PredictRequest):
    """Predict pore pressure."""
    try:
        mgr = get_manager()
        df = _df_from_request(req)
        preds = mgr.predict_pressure(df)
        if preds is None:
            raise HTTPException(status_code=422, detail="Pressure prediction unavailable")
        return PressureResponse(predictions=preds.tolist())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/all", response_model=AllPredictionsResponse)
async def predict_all(req: PredictRequest):
    """Run all three predictions and return a combined report."""
    result = AllPredictionsResponse()
    try:
        mgr = get_manager()
        df = _df_from_request(req)

        try:
            p = mgr.predict_porosity(df)
            result.porosity = PorosityResponse(predictions=p.tolist())
        except Exception as e:
            logger.warning("Porosity prediction failed: %s", e)

        try:
            preds, proba = mgr.predict_fluid(df)
            result.fluid = FluidResponse(
                predictions=preds.tolist(),
                probabilities=proba.tolist(),
                classes=["Background", "Potential Reservoir", "Pay Zone"],
            )
        except Exception as e:
            logger.warning("Fluid prediction failed: %s", e)

        try:
            p = mgr.predict_pressure(df)
            if p is not None:
                result.pressure = PressureResponse(predictions=p.tolist())
        except Exception as e:
            logger.warning("Pressure prediction failed: %s", e)

    except RuntimeError as e:
        detail = f"Models not ready: {e} — run /api/v1/health before predictions"
        raise HTTPException(status_code=503, detail=detail)

    return result
