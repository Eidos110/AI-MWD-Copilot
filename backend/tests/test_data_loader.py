import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.data_loader import load_data
from src.config import DATA_PATH, DISPLAY_COLS


class TestDataLoader:
    """Test suite for data_loader module"""
    
    def test_load_data_success(self):
        """Test successful data loading"""
        df = load_data()
        
        # Verify it returns a DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Verify it's not empty
        assert len(df) > 0
        
        # Verify essential columns exist
        assert 'DEPTH' in df.columns
        
    def test_load_data_columns(self):
        """Test that essential columns are present"""
        df = load_data()
        
        # Check DEPTH column
        assert 'DEPTH' in df.columns
        assert df['DEPTH'].dtype in [np.float64, np.float32, np.int64, np.int32]
        
    def test_load_data_sorted(self):
        """Test that data is sorted by DEPTH"""
        df = load_data()
        
        # Verify data is sorted by DEPTH
        assert df['DEPTH'].is_monotonic_increasing
        
    def test_load_data_no_duplicates_in_index(self):
        """Test that index is properly reset"""
        df = load_data()
        
        # Verify index is sequential
        assert (df.index == range(len(df))).all()
        
    def test_load_data_depth_values(self):
        """Test that DEPTH values are reasonable"""
        df = load_data()
        
        # Check depth is positive
        assert (df['DEPTH'] >= 0).all()
        
        # Check no NaN in DEPTH
        assert df['DEPTH'].notna().all()


class TestDataLoaderEdgeCases:
    """Test edge cases for data loader"""
    
    def test_load_data_file_exists(self):
        """Test that data file exists at expected path"""
        assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"
        
    def test_load_data_multiple_calls(self):
        """Test that multiple loads return consistent data"""
        df1 = load_data()
        df2 = load_data()
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
    def test_load_data_shape_consistency(self):
        """Test that loaded data has consistent shape"""
        df = load_data()
        
        rows, cols = df.shape
        assert rows > 0, "DataFrame has no rows"
        assert cols > 0, "DataFrame has no columns"
