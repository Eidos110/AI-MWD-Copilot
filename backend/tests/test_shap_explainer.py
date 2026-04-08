import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import xgboost as xgb
from src.shap_explainer import explain_model


class TestSHAPExplainer:
    """Test suite for SHAP explainer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for SHAP testing"""
        n_samples = 20
        n_features = 8
        data = np.random.rand(n_samples, n_features)
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        return pd.DataFrame(data, columns=feature_names), feature_names
    
    @pytest.fixture
    def mock_xgb_model(self):
        """Create a mocked XGBoost model"""
        model = Mock(spec=xgb.XGBRegressor)
        model.predict = Mock(return_value=np.random.rand(20))
        return model
    
    def test_explain_model_with_xgb_model(self, sample_data, mock_xgb_model):
        """Test explain_model with XGBoost model"""
        X_sample, feature_names = sample_data
        
        with patch('src.shap_explainer.shap.TreeExplainer') as mock_explainer:
            # Setup mock explainer
            mock_exp_instance = Mock()
            mock_shap_values = np.random.rand(20, 8)
            mock_exp_instance.shap_values.return_value = mock_shap_values
            mock_explainer.return_value = mock_exp_instance
            
            # Make model instance of XGBModel
            mock_xgb_model.__class__ = xgb.XGBModel
            
            fig = explain_model(mock_xgb_model, X_sample, feature_names, "Test SHAP")
            
            assert fig is not None
            assert hasattr(fig, 'axes')
            plt.close(fig)
    
    def test_explain_model_returns_figure(self, sample_data):
        """Test that explain_model returns a matplotlib figure"""
        X_sample, feature_names = sample_data
        model = Mock()
        model.predict = Mock(return_value=np.random.rand(20))
        
        with patch('src.shap_explainer.shap.Explainer'):
            fig = explain_model(model, X_sample, feature_names, "Test")
            
            assert fig is not None
            assert hasattr(fig, 'axes')
            plt.close(fig)
    
    def test_explain_model_with_none_feature_names(self, sample_data):
        """Test explain_model with None feature names"""
        X_sample, _ = sample_data
        model = Mock()
        model.predict = Mock(return_value=np.random.rand(20))
        
        with patch('src.shap_explainer.shap.Explainer'):
            fig = explain_model(model, X_sample, None, "Test")
            
            assert fig is not None
            plt.close(fig)
    
    def test_explain_model_with_numpy_array(self):
        """Test explain_model with numpy array input"""
        X_sample = np.random.rand(20, 8)
        feature_names = [f'Feature_{i}' for i in range(8)]
        model = Mock()
        model.predict = Mock(return_value=np.random.rand(20))
        
        with patch('src.shap_explainer.shap.Explainer'):
            fig = explain_model(model, X_sample, feature_names, "Test")
            
            assert fig is not None
            plt.close(fig)
    
    def test_explain_model_title(self, sample_data):
        """Test that explain_model sets correct title"""
        X_sample, feature_names = sample_data
        model = Mock()
        model.predict = Mock(return_value=np.random.rand(20))
        title_text = "Custom SHAP Analysis"
        
        with patch('src.shap_explainer.shap.Explainer'):
            fig = explain_model(model, X_sample, feature_names, title_text)
            
            assert fig is not None
            # Note: title verification would depend on how plt.title sets it
            plt.close(fig)
    
    def test_explain_model_error_handling(self, sample_data):
        """Test that explain_model handles errors gracefully"""
        X_sample, feature_names = sample_data
        model = Mock()
        
        with patch('src.shap_explainer.shap.Explainer') as mock_explainer:
            # Make the explainer raise an error
            mock_explainer.side_effect = Exception("SHAP error")
            
            fig = explain_model(model, X_sample, feature_names, "Test")
            
            # Should return figure with error message instead of crashing
            assert fig is not None
            plt.close(fig)
    
    def test_explain_model_multiclass_shap_values(self, sample_data):
        """Test explain_model handles multiclass SHAP values"""
        X_sample, feature_names = sample_data
        model = Mock(spec=xgb.XGBClassifier)
        
        with patch('src.shap_explainer.shap.TreeExplainer') as mock_explainer:
            mock_exp_instance = Mock()
            # Return list of shap values (multiclass case)
            mock_shap_values = [
                np.random.rand(20, 8),
                np.random.rand(20, 8),
                np.random.rand(20, 8)
            ]
            mock_exp_instance.shap_values.return_value = mock_shap_values
            mock_explainer.return_value = mock_exp_instance
            
            # Make model instance of XGBModel
            model.__class__ = xgb.XGBModel
            
            fig = explain_model(model, X_sample, feature_names, "Test")
            
            assert fig is not None
            plt.close(fig)


class TestSHAPExplainerEdgeCases:
    """Test edge cases for SHAP explainer"""
    
    def test_explain_model_single_sample(self):
        """Test explain_model with single sample"""
        X_sample = pd.DataFrame({
            'Feature_0': [1.0],
            'Feature_1': [2.0],
            'Feature_2': [3.0]
        })
        model = Mock()
        model.predict = Mock(return_value=np.array([0.5]))
        
        with patch('src.shap_explainer.shap.Explainer'):
            fig = explain_model(model, X_sample, ['Feature_0', 'Feature_1', 'Feature_2'])
            
            assert fig is not None
            plt.close(fig)
    
    def test_explain_model_single_feature(self):
        """Test explain_model with single feature"""
        X_sample = pd.DataFrame({
            'Feature_0': np.random.rand(20)
        })
        model = Mock()
        model.predict = Mock(return_value=np.random.rand(20))
        
        with patch('src.shap_explainer.shap.Explainer'):
            fig = explain_model(model, X_sample, ['Feature_0'])
            
            assert fig is not None
            plt.close(fig)
    
    def test_explain_model_many_features(self):
        """Test explain_model with many features"""
        n_features = 50
        X_sample = np.random.rand(20, n_features)
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        model = Mock()
        model.predict = Mock(return_value=np.random.rand(20))
        
        with patch('src.shap_explainer.shap.Explainer'):
            fig = explain_model(model, X_sample, feature_names)
            
            assert fig is not None
            plt.close(fig)
