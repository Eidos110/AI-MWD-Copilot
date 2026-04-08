"""Data loading and validation module.

Loads well-logging data from CSV, validates structure, sorts by depth,
and computes missing target columns if not already present.
"""

import io
import pandas as pd
import logging
from backend.core.config import settings

logger = logging.getLogger(__name__)

DISPLAY_COLS = [
    "DEPTH",
    "Gamma Ray - Corrected gAPI",
    "Resistivity Phase - Corrected - 2MHz ohm.m",
    "Bulk Density - Compensated kg/m3",
    "Neutron Porosity (Sandstone) Euc",
    "PHI_COMBINED",
    "FLUID_CLASS",
    "PREDICTED_PORE_PRESSURE_PSI",
]

TARGETS = ["PHI_COMBINED", "FLUID_CLASS", "PREDICTED_PORE_PRESSURE_PSI"]


def load_data():
    """Load and validate default well-logging dataset."""
    from backend.services.targets import compute_all_targets

    try:
        df = pd.read_csv(str(settings.DATA_PATH))

        essential = ["DEPTH"] + [c for c in DISPLAY_COLS if c != "DEPTH"]
        missing = [col for col in essential if col not in df.columns]
        if missing:
            logger.warning(f"Missing essential columns: {missing}")

        df = df.sort_values("DEPTH").reset_index(drop=True)

        missing_targets = [t for t in TARGETS if t not in df.columns]
        if missing_targets:
            df = compute_all_targets(df, inplace=True)

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")


def load_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load data from uploaded file bytes (CSV or Excel).

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used to detect format).

    Returns:
        pd.DataFrame: Loaded and sorted data.
    """
    from backend.services.targets import compute_all_targets

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError(f"Unsupported file format: {filename}")

        if "DEPTH" not in df.columns:
            raise ValueError("Uploaded file must contain a 'DEPTH' column.")

        df = df.sort_values("DEPTH").reset_index(drop=True)

        try:
            df = compute_all_targets(df, inplace=False)
        except Exception as e:
            logger.warning(f"Could not compute derived targets: {e}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load uploaded file: {e}")


def validate_data(df: pd.DataFrame) -> dict:
    """Validate uploaded data structure.

    Returns:
        dict with keys: valid, missing_columns, warnings
    """
    warnings = []
    essential = ["DEPTH"]
    missing = [col for col in essential if col not in df.columns]

    recommended = [c for c in DISPLAY_COLS if c != "DEPTH"]
    missing_recommended = [col for col in recommended if col not in df.columns]

    if missing_recommended:
        warnings.append(f"Missing recommended columns: {missing_recommended}")

    if len(df) == 0:
        warnings.append("Dataset is empty")

    if "DEPTH" in df.columns and df["DEPTH"].duplicated().any():
        warnings.append("Duplicate depth values detected")

    return {
        "valid": len(missing) == 0 and len(df) > 0,
        "missing_columns": missing,
        "missing_recommended": missing_recommended,
        "warnings": warnings,
        "row_count": len(df),
        "columns": list(df.columns),
        "depth_range": {
            "min": float(df["DEPTH"].min()) if "DEPTH" in df.columns else None,
            "max": float(df["DEPTH"].max()) if "DEPTH" in df.columns else None,
        },
    }
