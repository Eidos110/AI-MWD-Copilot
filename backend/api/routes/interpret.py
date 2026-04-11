"""Interpretation API routes."""

import logging
import numpy as np

# Fix for deprecated np.int in newer numpy versions
if not hasattr(np, "int"):
    np.int = int

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from backend.services.interpreter import (
    interpret_zones,
    interpret_petrophysics,
    interpret_drilling,
    generate_interpretation_report,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class InterpretRequest(BaseModel):
    data: List[Dict[str, Any]]
    depth_range: Optional[Dict[str, float]] = None
    include_confidence: bool = True


def _df_from_request(req: InterpretRequest) -> pd.DataFrame:
    """Convert request data to DataFrame."""
    df = pd.DataFrame(req.data)
    if req.depth_range:
        mask = (df["DEPTH"] >= req.depth_range.get("min", -np.inf)) & (
            df["DEPTH"] <= req.depth_range.get("max", np.inf)
        )
        df = df[mask].copy()
    return df


@router.post("/interpret/zones")
async def interpret_zones_endpoint(req: InterpretRequest):
    """Generate zone interpretation from fluid predictions."""
    try:
        df = _df_from_request(req)
        if len(df) == 0:
            raise HTTPException(
                status_code=400, detail="No data in specified depth range"
            )

        fluid_predictions = []
        confidences = []

        if req.data and len(req.data) > 0:
            first_row = req.data[0]
            if "fluid" in first_row:
                fluid_predictions = [f.get("fluid", "Background") for f in req.data]
            elif "FLUID_CLASS" in first_row:
                fluid_predictions = [
                    f.get("FLUID_CLASS", "Background") for f in req.data
                ]

            if req.include_confidence and "confidence" in first_row:
                confidences = [f.get("confidence", 0.0) for f in req.data]

        if not fluid_predictions:
            fluid_predictions = ["Background"] * len(df)

        result = interpret_zones(df, fluid_predictions, None, confidences)
        return result

    except Exception as e:
        logger.error(f"Zone interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret/petrophysics")
async def interpret_petrophysics_endpoint(req: InterpretRequest):
    """Generate petrophysical interpretation."""
    try:
        df = _df_from_request(req)
        if len(df) == 0:
            raise HTTPException(
                status_code=400, detail="No data in specified depth range"
            )

        porosity_pred = None
        pressure_pred = None

        if req.data and len(req.data) > 0:
            first_row = req.data[0]
            if "porosity" in first_row:
                porosity_pred = [
                    p.get("porosity") for p in req.data if p.get("porosity") is not None
                ]
            elif "PHI_COMBINED" in first_row:
                porosity_pred = [
                    p.get("PHI_COMBINED")
                    for p in req.data
                    if p.get("PHI_COMBINED") is not None
                ]

            if "pressure" in first_row:
                pressure_pred = [
                    p.get("pressure") for p in req.data if p.get("pressure") is not None
                ]
            elif "PREDICTED_PORE_PRESSURE_PSI" in first_row:
                pressure_pred = [
                    p.get("PREDICTED_PORE_PRESSURE_PSI")
                    for p in req.data
                    if p.get("PREDICTED_PORE_PRESSURE_PSI") is not None
                ]

        result = interpret_petrophysics(df, porosity_pred, pressure_pred)
        return result

    except Exception as e:
        logger.error(f"Petrophysics interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret/drilling")
async def interpret_drilling_endpoint(req: InterpretRequest):
    """Generate drilling decision support."""
    try:
        df = _df_from_request(req)
        if len(df) == 0:
            raise HTTPException(
                status_code=400, detail="No data in specified depth range"
            )

        pressure_pred = None
        porosity_pred = None

        if req.data and len(req.data) > 0:
            first_row = req.data[0]
            if "pressure" in first_row:
                pressure_pred = [
                    p.get("pressure") for p in req.data if p.get("pressure") is not None
                ]
            elif "PREDICTED_PORE_PRESSURE_PSI" in first_row:
                pressure_pred = [
                    p.get("PREDICTED_PORE_PRESSURE_PSI")
                    for p in req.data
                    if p.get("PREDICTED_PORE_PRESSURE_PSI") is not None
                ]

            if "porosity" in first_row:
                porosity_pred = [
                    p.get("porosity") for p in req.data if p.get("porosity") is not None
                ]
            elif "PHI_COMBINED" in first_row:
                porosity_pred = [
                    p.get("PHI_COMBINED")
                    for p in req.data
                    if p.get("PHI_COMBINED") is not None
                ]

        result = interpret_drilling(df, pressure_pred, porosity_pred)
        return result

    except Exception as e:
        logger.error(f"Drilling interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpret/all")
async def interpret_all_endpoint(req: InterpretRequest):
    """Generate complete interpretation report."""
    try:
        df = _df_from_request(req)
        if len(df) == 0:
            raise HTTPException(
                status_code=400, detail="No data in specified depth range"
            )

        fluid_predictions = None
        porosity_predictions = None
        pressure_predictions = None

        if req.data and len(req.data) > 0:
            first_row = req.data[0]
            if "fluid" in first_row:
                fluid_predictions = [p.get("fluid") for p in req.data]
            elif "FLUID_CLASS" in first_row:
                fluid_predictions = [p.get("FLUID_CLASS") for p in req.data]

            if "porosity" in first_row:
                porosity_predictions = [p.get("porosity") for p in req.data]
            elif "PHI_COMBINED" in first_row:
                porosity_predictions = [p.get("PHI_COMBINED") for p in req.data]

            if "pressure" in first_row:
                pressure_predictions = [p.get("pressure") for p in req.data]
            elif "PREDICTED_PORE_PRESSURE_PSI" in first_row:
                pressure_predictions = [
                    p.get("PREDICTED_PORE_PRESSURE_PSI") for p in req.data
                ]

        report = generate_interpretation_report(
            df,
            fluid_predictions,
            None,
            porosity_predictions,
            pressure_predictions,
        )

        return report

    except Exception as e:
        logger.error(f"Full interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
