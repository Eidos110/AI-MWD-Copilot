# src/data_loader.py

import pandas as pd
from src.config import DATA_PATH, DISPLAY_COLS

def load_data():
    """Load and validate dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Ensure essential columns exist
        essential = ['DEPTH'] + [c for c in DISPLAY_COLS if c != 'DEPTH']
        missing = [col for col in essential if col not in df.columns]
        if missing:
            raise ValueError(f"Missing essential columns: {missing}")
            
        # Sort by depth
        df = df.sort_values('DEPTH').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"Failed to load  {str(e)}")