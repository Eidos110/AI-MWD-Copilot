"""Data quality assessment API routes."""

import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.services.data_quality import (
    analyze_missing_values,
    detect_outliers,
    compute_data_completeness,
    compute_sensor_health,
    generate_quality_report,
)
from backend.services.model_manager import (
    FEATURES_POROSITY,
    FEATURES_FLUID,
    FEATURES_PRESSURE,
)
from backend.services.data_loader import DISPLAY_COLS

logger = logging.getLogger(__name__)
router = APIRouter()


class QualityRequest(BaseModel):
    data: List[Dict[str, Any]]


class QualityResponse(BaseModel):
    missing_values: List[Dict[str, Any]]
    completeness: Dict[str, Any]
    sensor_health: List[Dict[str, Any]]
    summary: str


@router.post("/quality/report", response_model=QualityResponse)
async def get_quality_report(req: QualityRequest):
    """Generate comprehensive data quality report."""
    try:
        df = pd.DataFrame(req.data)
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")

        report = generate_quality_report(
            df,
            critical_cols=[c for c in DISPLAY_COLS if c in df.columns],
            feature_groups={
                "Porosity": [f for f in FEATURES_POROSITY if f in df.columns],
                "Fluid": [f for f in FEATURES_FLUID if f in df.columns],
                "Pressure": [f for f in FEATURES_PRESSURE if f in df.columns],
            },
        )

        missing_list = report["missing_values"].to_dict(orient="records")
        health_list = report["sensor_health"].to_dict(orient="records")

        return {
            "missing_values": missing_list,
            "completeness": report["completeness"],
            "sensor_health": health_list,
            "summary": report["summary"],
        }
    except Exception as e:
        logger.error(f"Quality report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
