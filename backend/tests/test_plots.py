import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

# Mock config before importing plot_well_logs
with patch.dict('sys.modules', {'src.config': Mock()}):
    from src.plots import plot_well_logs

# Define column names for tests
COL_DEPTH = 'DEPTH'
COL_GAMMA_RAY = 'Gamma Ray - Corrected gAPI'
COL_RESISTIVITY = 'Resistivity Phase - Corrected - 2MHz ohm.m'
COL_POROSITY = 'PHI_COMBINED'
COL_FLUID_CLASS = 'FLUID_CLASS'
COL_PORE_PRESSURE = 'PREDICTED_PORE_PRESSURE_PSI'
COL_WOB = 'Weight On Bit N'
COL_TORQUE = 'Surface Torque Average N.m'


class TestPlotWellLogs:
    """Test suite for plot_well_logs function"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing"""
        n_points = 100
        data = {
            COL_DEPTH: np.linspace(2000, 2100, n_points),
            COL_GAMMA_RAY: np.random.rand(n_points) * 150,
            COL_RESISTIVITY: np.random.rand(n_points) * 100 + 1,
            COL_POROSITY: np.random.rand(n_points) * 0.3,
            COL_FLUID_CLASS: np.random.choice(['Oil', 'Water', 'Gas'], n_points),
            COL_PORE_PRESSURE: np.random.rand(n_points) * 10000 + 2000,
            COL_WOB: np.random.rand(n_points) * 100000,
            COL_TORQUE: np.random.rand(n_points) * 50000,
        }
        return pd.DataFrame(data)
    
    def test_plot_well_logs_returns_figure(self, sample_dataframe):
        """Test that plot_well_logs returns a matplotlib figure"""
        fig = plot_well_logs(sample_dataframe)
        
        assert fig is not None
        assert hasattr(fig, 'axes')
        plt.close(fig)
    
    def test_plot_well_logs_creates_multiple_tracks(self, sample_dataframe):
        """Test that plot creates multiple subplots (tracks)"""
        fig = plot_well_logs(sample_dataframe)
        
        # Should have 6 tracks
        assert len(fig.axes) == 6
        plt.close(fig)
    
    def test_plot_well_logs_with_depth_range(self, sample_dataframe):
        """Test plot with depth range filtering"""
        depth_range = (2020, 2080)
        fig = plot_well_logs(sample_dataframe, depth_range=depth_range)
        
        assert fig is not None
        assert len(fig.axes) == 6
        plt.close(fig)
    
    def test_plot_well_logs_with_predictions(self, sample_dataframe):
        """Test plot with prediction data"""
        predictions = {
            'phi_pred': np.random.rand(len(sample_dataframe)) * 0.3
        }
        fig = plot_well_logs(sample_dataframe, predictions=predictions)
        
        assert fig is not None
        assert len(fig.axes) == 6
        plt.close(fig)
    
    def test_plot_well_logs_empty_depth_range(self, sample_dataframe):
        """Test plot with empty depth range"""
        depth_range = (3000, 4000)  # No data in this range
        fig = plot_well_logs(sample_dataframe, depth_range=depth_range)
        
        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)
    
    def test_plot_well_logs_missing_columns(self):
        """Test plot with missing optional columns"""
        df = pd.DataFrame({
            COL_DEPTH: np.linspace(2000, 2100, 10),
            'SomeOtherColumn': np.random.rand(10)
        })
        
        fig = plot_well_logs(df)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_well_logs_with_nan_values(self, sample_dataframe):
        """Test plot handles NaN values"""
        # Introduce some NaN values
        sample_dataframe.loc[10:20, COL_GAMMA_RAY] = np.nan
        sample_dataframe.loc[30:40, COL_RESISTIVITY] = np.nan
        
        fig = plot_well_logs(sample_dataframe)
        
        assert fig is not None
        assert len(fig.axes) == 6
        plt.close(fig)
    
    def test_plot_well_logs_title(self, sample_dataframe):
        """Test that plot has appropriate title"""
        fig = plot_well_logs(sample_dataframe)
        
        # Check that figure has a title
        assert fig._suptitle is not None
        plt.close(fig)


class TestPlotEdgeCases:
    """Test edge cases for plotting functions"""
    
    def test_plot_empty_dataframe(self):
        """Test plot with empty dataframe"""
        df = pd.DataFrame({
            COL_DEPTH: []
        })
        
        fig = plot_well_logs(df)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_single_row(self):
        """Test plot with single row of data"""
        df = pd.DataFrame({
            COL_DEPTH: [2000],
            COL_GAMMA_RAY: [100],
            COL_RESISTIVITY: [50],
        })
        
        fig = plot_well_logs(df)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_all_nan_column(self):
        """Test plot with column containing all NaN"""
        df = pd.DataFrame({
            COL_DEPTH: np.linspace(2000, 2100, 10),
            COL_GAMMA_RAY: np.full(10, np.nan),
            COL_RESISTIVITY: np.random.rand(10) * 100,
        })
        
        fig = plot_well_logs(df)
        
        assert fig is not None
        plt.close(fig)
