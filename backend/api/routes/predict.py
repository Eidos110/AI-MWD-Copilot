"""Prediction API routes."""

import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from backend.services.model_manager import ModelManager
from backend.services.predictions import (
    compute_prediction_confidence,
    compute_prediction_intervals,
    create_prediction_report,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global model manager reference
_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        from backend.app import get_model_manager

        _manager = get_model_manager()
    return _manager


class DataRow(BaseModel):
    DEPTH: float
    model_config = {"extra": "allow"}


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
        if len(df) == 0:
            raise HTTPException(
                status_code=400, detail="No data in specified depth range"
            )

        predictions = mgr.predict_porosity(df)
        result = {"predictions": predictions.tolist()}

        if req.include_confidence:
            X = mgr._safe_select(
                df,
                mgr.porosity_model.get_booster().feature_names
                if hasattr(mgr.porosity_model, "get_booster")
                else [],
                "porosity",
            )
            if not X.empty:
                X = X.fillna(X.mean())
                conf = compute_prediction_confidence(mgr.porosity_model, X, predictions)
                result["confidence"] = conf.tolist()
                lower, upper = compute_prediction_intervals(predictions, conf)
                result["intervals"] = {"lower": lower.tolist(), "upper": upper.tolist()}

        return result
    except Exception as e:
        logger.error(f"Porosity prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/fluid", response_model=FluidResponse)
async def predict_fluid(req: PredictRequest):
    """Predict fluid type and class probabilities."""
    try:
        mgr = get_manager()
        df = _df_from_request(req)
        if len(df) == 0:
            raise HTTPException(
                status_code=400, detail="No data in specified depth range"
            )

        fluid_classes, fluid_probs = mgr.predict_fluid(df)
        return {
            "predictions": fluid_classes.tolist(),
            "probabilities": fluid_probs.tolist(),
            "classes": list(mgr.fluid_encoder.classes_),
        }
    except Exception as e:
        logger.error(f"Fluid prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/pressure", response_model=PressureResponse)
async def predict_pressure(req: PredictRequest):
    """Predict pore pressure."""
    try:
        mgr = get_manager()
        df = _df_from_request(req)
        if len(df) == 0:
            raise HTTPException(
                status_code=400, detail="No data in specified depth range"
            )

        pp_pred = mgr.predict_pressure(df)
        if pp_pred is None:
            raise HTTPException(
                status_code=500, detail="Pressure prediction unavailable"
            )

        result = {"predictions": pp_pred.tolist()}

        if req.include_confidence:
            X = mgr._safe_select(df, [], "pressure", force_full=True)
            if not X.empty:
                X = X.fillna(X.mean()).fillna(0)
                expected = mgr.pressure_model.get_booster().feature_names
                for f in expected:
                    if f not in X.columns:
                        X[f] = 0
                X = X[expected]
                conf = compute_prediction_confidence(mgr.pressure_model, X, pp_pred)
                result["confidence"] = conf.tolist()
                lower, upper = compute_prediction_intervals(pp_pred, conf)
                result["intervals"] = {"lower": lower.tolist(), "upper": upper.tolist()}

        return result
    except Exception as e:
        logger.error(f"Pressure prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/all", response_model=AllPredictionsResponse)
async def predict_all(req: PredictRequest):
    """Run all three prediction models at once."""
    try:
        mgr = get_manager()
        df = _df_from_request(req)
        if len(df) == 0:
            raise HTTPException(
                status_code=400, detail="No data in specified depth range"
            )

        response = {}
        predictions = {}
        confidences = {}

        # Porosity
        try:
            phi_pred = mgr.predict_porosity(df)
            predictions["phi_pred"] = phi_pred
            response["porosity"] = {"predictions": phi_pred.tolist()}
            if req.include_confidence:
                X = mgr._safe_select(df, [], "porosity")
                if not X.empty:
                    X = X.fillna(X.mean())
                    conf = compute_prediction_confidence(
                        mgr.porosity_model, X, phi_pred
                    )
                    confidences["phi_conf"] = conf
                    response["porosity"]["confidence"] = conf.tolist()
                    lower, upper = compute_prediction_intervals(phi_pred, conf)
                    response["porosity"]["intervals"] = {
                        "lower": lower.tolist(),
                        "upper": upper.tolist(),
                    }
        except Exception as e:
            logger.warning(f"Porosity prediction failed: {e}")

        # Fluid
        try:
            fluid_classes, fluid_probs = mgr.predict_fluid(df)
            predictions["fluid_pred"] = fluid_classes
            predictions["fluid_prob"] = fluid_probs
            response["fluid"] = {
                "predictions": fluid_classes.tolist(),
                "probabilities": fluid_probs.tolist(),
                "classes": list(mgr.fluid_encoder.classes_),
            }
        except Exception as e:
            logger.warning(f"Fluid prediction failed: {e}")

        # Pressure
        try:
            pp_pred = mgr.predict_pressure(df)
            if pp_pred is not None:
                predictions["pp_pred"] = pp_pred
                response["pressure"] = {"predictions": pp_pred.tolist()}
                if req.include_confidence:
                    X = mgr._safe_select(df, [], "pressure", force_full=True)
                    if not X.empty:
                        X = X.fillna(X.mean()).fillna(0)
                        expected = mgr.pressure_model.get_booster().feature_names
                        for f in expected:
                            if f not in X.columns:
                                X[f] = 0
                        X = X[expected]
                        conf = compute_prediction_confidence(
                            mgr.pressure_model, X, pp_pred
                        )
                        confidences["pp_conf"] = conf
                        response["pressure"]["confidence"] = conf.tolist()
                        lower, upper = compute_prediction_intervals(pp_pred, conf)
                        response["pressure"]["intervals"] = {
                            "lower": lower.tolist(),
                            "upper": upper.tolist(),
                        }
        except Exception as e:
            logger.warning(f"Pressure prediction failed: {e}")

        # Generate report
        if predictions:
            report_df = create_prediction_report(df, predictions, confidences)
            response["report"] = report_df.to_dict(orient="records")

        return response
    except Exception as e:
        logger.error(f"All predictions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
