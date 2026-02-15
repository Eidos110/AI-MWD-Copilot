# src/data_loader.py
"""
Data loading and validation module.

Loads well-logging data from CSV, validates structure, sorts by depth,
and computes missing target columns (PHI_COMBINED, FLUID_CLASS, PREDICTED_PORE_PRESSURE_PSI)
if not already present in the dataset.

Functions:
    load_data(): Load CSV, validate essential columns, sort, and compute targets.
"""

import pandas as pd
from src.config import DATA_PATH, DISPLAY_COLS, TARGETS
from src.targets import compute_all_targets

def load_data():
    """Load and validate well-logging dataset.
    
    Steps:
    1. Load CSV from DATA_PATH.
    2. Verify essential columns exist (DEPTH + DISPLAY_COLS).
    3. Sort by DEPTH in ascending order.
    4. Compute target columns (PHI_COMBINED, FLUID_CLASS, PREDICTED_PORE_PRESSURE_PSI)
       if any are missing.
    
    Returns:
        pd.DataFrame: Validated and processed dataset, sorted by depth.
    
    Raises:
        RuntimeError: If data file cannot be loaded or essential columns are missing.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Ensure essential columns exist
        essential = ['DEPTH'] + [c for c in DISPLAY_COLS if c != 'DEPTH']
        missing = [col for col in essential if col not in df.columns]
        if missing:
            raise ValueError(f"Missing essential columns: {missing}")
            
        # Sort by depth
        df = df.sort_values('DEPTH').reset_index(drop=True)

        # If any expected target columns are missing, compute them defensively
        missing_targets = [t for t in TARGETS if t not in df.columns]
        if missing_targets:
            # compute_all_targets will be defensive about missing input columns
            df = compute_all_targets(df, inplace=True)

        return df
    
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {str(e)}")