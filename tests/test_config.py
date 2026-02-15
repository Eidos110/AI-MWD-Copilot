import pytest
import os
from src import config


class TestConfigPaths:
    """Test suite for configuration paths"""
    
    def test_root_dir_exists(self):
        """Test that ROOT_DIR exists"""
        assert os.path.isdir(config.ROOT_DIR)
        
    def test_data_dir_exists(self):
        """Test that DATA_DIR exists"""
        assert os.path.isdir(config.DATA_DIR)
        
    def test_model_dir_exists(self):
        """Test that MODEL_DIR exists"""
        assert os.path.isdir(config.MODEL_DIR)
        
    def test_data_path_defined(self):
        """Test that DATA_PATH is properly defined"""
        assert hasattr(config, 'DATA_PATH')
        assert config.DATA_PATH is not None
        assert isinstance(config.DATA_PATH, str)


class TestConfigFeatures:
    """Test suite for feature definitions"""
    
    def test_features_porosity_defined(self):
        """Test that porosity features are defined"""
        assert hasattr(config, 'FEATURES_POROSITY')
        assert isinstance(config.FEATURES_POROSITY, list)
        assert len(config.FEATURES_POROSITY) > 0
        assert 'DEPTH' in config.FEATURES_POROSITY
        
    def test_features_fluid_defined(self):
        """Test that fluid features are defined"""
        assert hasattr(config, 'FEATURES_FLUID')
        assert isinstance(config.FEATURES_FLUID, list)
        assert len(config.FEATURES_FLUID) > 0
        assert 'DEPTH' in config.FEATURES_FLUID
        
    def test_features_pressure_defined(self):
        """Test that pressure features are defined"""
        assert hasattr(config, 'FEATURES_PRESSURE')
        assert isinstance(config.FEATURES_PRESSURE, list)
        assert len(config.FEATURES_PRESSURE) > 0
        
    def test_minimal_features_defined(self):
        """Test that minimal features are defined"""
        assert hasattr(config, 'MINIMAL_FEATURES')
        assert isinstance(config.MINIMAL_FEATURES, list)
        assert len(config.MINIMAL_FEATURES) > 0


class TestConfigConsistency:
    """Test consistency of configuration"""
    
    def test_feature_lists_no_duplicates(self):
        """Test that feature lists have no duplicates"""
        assert len(config.FEATURES_POROSITY) == len(set(config.FEATURES_POROSITY))
        assert len(config.FEATURES_FLUID) == len(set(config.FEATURES_FLUID))
        assert len(config.FEATURES_PRESSURE) == len(set(config.FEATURES_PRESSURE))
        
    def test_features_are_strings(self):
        """Test that all features are strings"""
        assert all(isinstance(f, str) for f in config.FEATURES_POROSITY)
        assert all(isinstance(f, str) for f in config.FEATURES_FLUID)
        assert all(isinstance(f, str) for f in config.FEATURES_PRESSURE)
        assert all(isinstance(f, str) for f in config.MINIMAL_FEATURES)
