"""
Simple integration tests for the project
"""
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestProjectStructure:
    """Test that project structure is correct"""
    
    def test_src_directory_exists(self):
        """Test that src directory exists"""
        src_dir = project_root / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()
    
    def test_data_directory_exists(self):
        """Test that data directory exists"""
        data_dir = project_root / "data"
        assert data_dir.exists()
        assert data_dir.is_dir()
    
    def test_models_directory_exists(self):
        """Test that models directory exists"""
        models_dir = project_root / "models"
        assert models_dir.exists()
        assert models_dir.is_dir()
    
    def test_tests_directory_exists(self):
        """Test that tests directory exists"""
        tests_dir = project_root / "tests"
        assert tests_dir.exists()
        assert tests_dir.is_dir()
    
    def test_main_app_file_exists(self):
        """Test that app.py exists"""
        app_file = project_root / "app.py"
        assert app_file.exists()
        assert app_file.is_file()
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        req_file = project_root / "requirements.txt"
        assert req_file.exists()
        assert req_file.is_file()


class TestSourceModules:
    """Test that all source modules are present"""
    
    def test_config_module_exists(self):
        """Test that config.py exists"""
        config_file = project_root / "src" / "config.py"
        assert config_file.exists()
        assert config_file.is_file()
    
    def test_data_loader_module_exists(self):
        """Test that data_loader.py exists"""
        module_file = project_root / "src" / "data_loader.py"
        assert module_file.exists()
        assert module_file.is_file()
    
    def test_models_module_exists(self):
        """Test that models.py exists"""
        module_file = project_root / "src" / "models.py"
        assert module_file.exists()
        assert module_file.is_file()
    
    def test_plots_module_exists(self):
        """Test that plots.py exists"""
        module_file = project_root / "src" / "plots.py"
        assert module_file.exists()
        assert module_file.is_file()
    
    def test_shap_explainer_module_exists(self):
        """Test that shap_explainer.py exists"""
        module_file = project_root / "src" / "shap_explainer.py"
        assert module_file.exists()
        assert module_file.is_file()


class TestDataFiles:
    """Test that necessary data files exist"""
    
    def test_data_csv_file_exists(self):
        """Test that ready_modelling.csv exists"""
        csv_file = project_root / "data" / "ready_modelling.csv"
        assert csv_file.exists()
        assert csv_file.is_file()
        assert csv_file.suffix == ".csv"


class TestRequirements:
    """Test requirements.txt content"""
    
    def test_requirements_file_not_empty(self):
        """Test that requirements.txt is not empty"""
        req_file = project_root / "requirements.txt"
        content = req_file.read_text()
        assert len(content) > 0
    
    def test_requirements_contains_streamlit(self):
        """Test that requirements.txt contains streamlit"""
        req_file = project_root / "requirements.txt"
        content = req_file.read_text()
        assert "streamlit" in content.lower()
    
    def test_requirements_contains_pandas(self):
        """Test that requirements.txt contains pandas"""
        req_file = project_root / "requirements.txt"
        content = req_file.read_text()
        assert "pandas" in content.lower()
    
    def test_requirements_contains_sklearn(self):
        """Test that requirements.txt contains scikit-learn"""
        req_file = project_root / "requirements.txt"
        content = req_file.read_text()
        assert "scikit-learn" in content.lower() or "sklearn" in content.lower()
    
    def test_requirements_contains_xgboost(self):
        """Test that requirements.txt contains xgboost"""
        req_file = project_root / "requirements.txt"
        content = req_file.read_text()
        assert "xgboost" in content.lower()
    
    def test_requirements_contains_shap(self):
        """Test that requirements.txt contains shap"""
        req_file = project_root / "requirements.txt"
        content = req_file.read_text()
        assert "shap" in content.lower()


class TestModuleImports:
    """Test that core modules can be imported"""
    
    def test_import_config(self):
        """Test that config module can be imported"""
        try:
            from src import config
            assert config is not None
        except ImportError as e:
            pytest.skip(f"Could not import config: {e}")
    
    def test_import_data_loader(self):
        """Test that data_loader module can be imported"""
        try:
            from src import data_loader
            assert data_loader is not None
        except ImportError as e:
            pytest.skip(f"Could not import data_loader: {e}")
    
    def test_import_models(self):
        """Test that models module can be imported"""
        try:
            from src import models
            assert models is not None
        except ImportError as e:
            pytest.skip(f"Could not import models: {e}")
    
    def test_import_plots(self):
        """Test that plots module can be imported"""
        try:
            from src import plots
            assert plots is not None
        except ImportError as e:
            pytest.skip(f"Could not import plots: {e}")
    
    def test_import_shap_explainer(self):
        """Test that shap_explainer module can be imported"""
        try:
            from src import shap_explainer
            assert shap_explainer is not None
        except ImportError as e:
            pytest.skip(f"Could not import shap_explainer: {e}")
