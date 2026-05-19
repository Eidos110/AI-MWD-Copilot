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
# Lazy-load DISPLAY_COLS so route module is importable even before
# data_loader or model_manager are ready.
from backend.services.data_loader import DISPLAY_COLS

logger = logging.getLogger(__name__)
router = APIRouter()

# Feature sets duplicated here to avoid a hard import of model_manager
# at module level. The single source of truth is
#   backend/services/model_manager.py
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
                "Porosity": [f for f in _POROSITY_FEATURES if f in df.columns],
                "Fluid": [f for f in _FLUID_FEATURES if f in df.columns],
                "Pressure": [f for f in _PRESSURE_FEATURES if f in df.columns],
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
        logger.error("Quality report failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
