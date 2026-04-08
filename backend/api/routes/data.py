"""Data upload and retrieval API routes."""

import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.services.data_loader import (
    load_data,
    load_uploaded_file,
    validate_data,
    DISPLAY_COLS,
)
from backend.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Cache for default data
_default_data: Optional[pd.DataFrame] = None
_uploaded_data: Optional[pd.DataFrame] = None


def _get_default_data() -> pd.DataFrame:
    global _default_data
    if _default_data is None:
        _default_data = load_data()
    return _default_data


class SampleDataResponse(BaseModel):
    columns: List[str]
    rows: int
    data: List[Dict[str, Any]]
    depth_range: Dict[str, float]


class UploadResponse(BaseModel):
    filename: str
    columns: List[str]
    rows: int
    preview: List[Dict[str, Any]]
    depth_range: Dict[str, float]
    validation: Dict[str, Any]


class ValidateResponse(BaseModel):
    valid: bool
    missing_columns: List[str]
    missing_recommended: List[str]
    warnings: List[str]
    row_count: int
    columns: List[str]
    depth_range: Dict[str, Any]


@router.get("/data/sample", response_model=SampleDataResponse)
async def get_sample_data(
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
):
    """Get the default dataset with optional depth filtering."""
    try:
        df = _get_default_data()
        if min_depth is not None:
            df = df[df["DEPTH"] >= min_depth]
        if max_depth is not None:
            df = df[df["DEPTH"] <= max_depth]

        # Return ALL columns so ML models can use all features
        preview = df.to_dict(orient="records")

        return {
            "columns": list(df.columns),
            "rows": len(df),
            "data": preview,
            "depth_range": {
                "min": float(df["DEPTH"].min()),
                "max": float(df["DEPTH"].max()),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/upload", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """Upload CSV or Excel file."""
    global _uploaded_data
    try:
        contents = await file.read()
        df = load_uploaded_file(contents, file.filename or "")
        _uploaded_data = df

        validation = validate_data(df)
        preview = df.head(10).to_dict(orient="records")

        return {
            "filename": file.filename or "unknown",
            "columns": list(df.columns),
            "rows": len(df),
            "preview": preview,
            "depth_range": {
                "min": float(df["DEPTH"].min()),
                "max": float(df["DEPTH"].max()),
            },
            "validation": validation,
        }
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/validate", response_model=ValidateResponse)
async def validate_uploaded_data(file: UploadFile = File(...)):
    """Validate uploaded data structure without storing."""
    try:
        contents = await file.read()
        df = load_uploaded_file(contents, file.filename or "")
        result = validate_data(df)
        return result
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/columns")
async def get_columns():
    """Get all available column names from the default dataset."""
    try:
        df = _get_default_data()
        return {"columns": list(df.columns)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
