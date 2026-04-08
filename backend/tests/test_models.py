import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.models import ModelManager
from src.config import FEATURES_POROSITY, FEATURES_FLUID, FEATURES_PRESSURE, MINIMAL_FEATURES


class TestModelManagerInit:
    """Test ModelManager initialization"""
    
    @patch('src.models.joblib.load')
    def test_model_manager_init(self, mock_load):
        """Test ModelManager initialization with mocked models"""
        # Setup mocks
        mock_porosity_model = Mock()
        mock_fluid_model = Mock()
        mock_pressure_model = Mock()
        mock_encoder = Mock()
        
        mock_load.side_effect = [
            mock_porosity_model,
            mock_fluid_model,
            mock_pressure_model,
            mock_encoder
        ]
        
        # Initialize ModelManager
        manager = ModelManager()
        
        # Verify models are loaded
        assert manager.porosity_model is not None
        assert manager.fluid_model is not None
        assert manager.pressure_model is not None
        assert manager.fluid_encoder is not None
        assert mock_load.call_count == 4


class TestModelManagerPredictions:
    """Test ModelManager prediction methods"""
    
    @pytest.fixture
    def mock_manager(self):
        """Create a mocked ModelManager for testing"""
        with patch('src.models.joblib.load'):
            manager = ModelManager()
            # Mock the individual models
            manager.porosity_model = Mock()
            manager.fluid_model = Mock()
            manager.pressure_model = Mock()
            manager.fluid_encoder = Mock()
            return manager
    
    def create_test_dataframe(self, features, n_rows=10):
        """Helper to create test dataframe with specified features"""
        data = {}
        for feat in features:
            if feat == 'DEPTH':
                data[feat] = np.linspace(2000, 2100, n_rows)
            else:
                data[feat] = np.random.rand(n_rows) * 100
        return pd.DataFrame(data)
    
    def test_predict_porosity(self, mock_manager):
        """Test porosity prediction"""
        # Setup
        mock_manager.porosity_model.predict.return_value = np.array([0.2, 0.25, 0.3] * 4)[:10]
        
        df = self.create_test_dataframe(FEATURES_POROSITY, n_rows=10)
        
        # Execute
        predictions = mock_manager.predict_porosity(df)
        
        # Verify
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(df)
        mock_manager.porosity_model.predict.assert_called_once()
    
    def test_predict_fluid(self, mock_manager):
        """Test fluid type prediction"""
        # Setup
        mock_manager.fluid_model.predict.return_value = np.array([0, 1, 0] * 4)[:10]
        mock_manager.fluid_model.predict_proba.return_value = np.random.rand(10, 3)
        mock_manager.fluid_encoder.inverse_transform.return_value = np.array(['Oil', 'Water', 'Gas'] * 4)[:10]
        
        df = self.create_test_dataframe(FEATURES_FLUID, n_rows=10)
        
        # Execute
        fluid_types, probas = mock_manager.predict_fluid(df)
        
        # Verify
        assert len(fluid_types) == len(df)
        assert probas.shape[0] == len(df)
        assert probas.shape[1] == 3  # 3 classes
        mock_manager.fluid_model.predict.assert_called_once()
        mock_manager.fluid_model.predict_proba.assert_called_once()
    
    def test_predict_pressure(self, mock_manager):
        """Test pore pressure prediction"""
        # Setup
        mock_model = Mock()
        mock_booster = Mock()
        mock_booster.feature_names = FEATURES_PRESSURE
        mock_model.get_booster.return_value = mock_booster
        mock_model.predict.return_value = np.array([5000 + i*10 for i in range(10)])
        
        mock_manager.pressure_model = mock_model
        
        df = self.create_test_dataframe(FEATURES_PRESSURE, n_rows=10)
        
        # Execute
        predictions = mock_manager.predict_pressure(df)
        
        # Verify
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(df)
        assert all(pred >= 0 for pred in predictions)
    
    def test_safe_select_with_all_features(self, mock_manager):
        """Test _safe_select with all features available"""
        df = self.create_test_dataframe(FEATURES_POROSITY)
        
        # Execute
        selected = mock_manager._safe_select(df, FEATURES_POROSITY, "porosity")
        
        # Verify
        assert isinstance(selected, pd.DataFrame)
        assert all(feat in selected.columns for feat in FEATURES_POROSITY)
    
    def test_safe_select_with_missing_features(self, mock_manager):
        """Test _safe_select with missing features"""
        # Create dataframe with only minimal features
        df = self.create_test_dataframe(MINIMAL_FEATURES)
        
        # Execute
        selected = mock_manager._safe_select(df, FEATURES_POROSITY, "porosity")
        
        # Verify
        assert isinstance(selected, pd.DataFrame)
        assert len(selected) > 0
    
    def test_safe_select_force_full(self, mock_manager):
        """Test _safe_select with force_full=True"""
        df = self.create_test_dataframe(['DEPTH', 'GR'], n_rows=5)
        
        # Execute
        selected = mock_manager._safe_select(df, FEATURES_POROSITY, "porosity", force_full=True)
        
        # Verify
        assert all(feat in selected.columns for feat in FEATURES_POROSITY)
        assert len(selected) == len(df)


class TestModelManagerDataHandling:
    """Test data handling in ModelManager"""
    
    @patch('src.models.joblib.load')
    def test_nan_handling_in_predictions(self, mock_load):
        """Test that NaN values are properly handled"""
        with patch('src.models.joblib.load'):
            manager = ModelManager()
            manager.porosity_model = Mock()
            manager.porosity_model.predict.return_value = np.array([0.2] * 5)
            
            # Create dataframe with NaN values
            data = {
                'DEPTH': [2000, 2010, 2020, 2030, 2040],
                'Feature1': [1.0, np.nan, 3.0, np.nan, 5.0],
                'Feature2': [10, 20, 30, 40, 50],
            }
            for feat in FEATURES_POROSITY:
                if feat not in data:
                    data[feat] = np.random.rand(5) * 100
            
            df = pd.DataFrame(data)
            
            # Execute - should handle NaNs
            predictions = manager.predict_porosity(df)
            
            # Verify
            assert len(predictions) == len(df)
            assert not np.any(np.isnan(predictions))
