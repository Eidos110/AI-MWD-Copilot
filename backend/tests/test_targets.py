"""
Unit tests for src/targets.py module.

Tests cover:
- Porosity composite calculation (Wyllie density + neutron)
- Rule-based fluid classification
- Simplified pore pressure estimation (Rehm & McClendon style)
"""
import pytest
import pandas as pd
import numpy as np
from src.targets import (
    _phi_from_density,
    compute_phi_combined,
    compute_fluid_class,
    compute_pore_pressure,
    compute_all_targets
)


class TestPhiFromDensity:
    """Test Wyllie-style porosity calculation from bulk density."""

    def test_zero_porosity(self):
        """Test dense rock (zero porosity)."""
        rho_bulk = 2650.0  # Matrix density
        phi = _phi_from_density(rho_bulk)
        assert phi == 0.0

    def test_full_porosity(self):
        """Test fully fluid-filled (100% porosity)."""
        rho_bulk = 1000.0  # Fluid density
        phi = _phi_from_density(rho_bulk)
        assert phi == 1.0

    def test_moderate_porosity(self):
        """Test intermediate porosity."""
        rho_bulk = 2200.0  # Typical sandstone
        phi = _phi_from_density(rho_bulk, rho_matrix=2650.0, rho_fluid=1000.0)
        expected = (2650.0 - 2200.0) / (2650.0 - 1000.0)  # ~0.276
        assert abs(phi - expected) < 0.01

    def test_clipped_to_range(self):
        """Test that porosity is clipped to [0, 1]."""
        # Very low density (physically impossible)
        phi_high = _phi_from_density(500.0)
        assert phi_high == 1.0  # Clipped to max

        # Very high density (physically impossible)
        phi_low = _phi_from_density(3000.0)
        assert phi_low == 0.0  # Clipped to min

    def test_nan_handling(self):
        """Test handling of NaN values."""
        rho_bulk = np.array([2200.0, np.nan, 2300.0])
        phi = _phi_from_density(rho_bulk)
        assert np.isfinite(phi[0]) and np.isfinite(phi[2])
        assert np.isnan(phi[1])


class TestComputePhiCombined:
    """Test composite porosity calculation."""

    def test_single_density_log(self):
        """Test porosity from density log only."""
        df = pd.DataFrame({
            'Bulk Density - Compensated kg/m3': [2200.0, 2300.0, 2400.0]
        })
        phi = compute_phi_combined(df)
        assert len(phi) == 3
        assert all(0 <= p <= 1 for p in phi if np.isfinite(p))

    def test_single_neutron_log(self):
        """Test porosity from neutron log only."""
        df = pd.DataFrame({
            'Neutron Porosity (Sandstone) Euc': [0.2, 0.25, 0.3]
        })
        phi = compute_phi_combined(df)
        assert len(phi) == 3
        assert all(np.isclose(p, expected) for p, expected in
                  zip(phi, [0.2, 0.25, 0.3]))

    def test_percent_neutron_conversion(self):
        """Test conversion of neutron porosity from percent to fraction."""
        df = pd.DataFrame({
            'Neutron Porosity (Sandstone) Euc': [20.0, 25.0, 30.0]  # percent
        })
        phi = compute_phi_combined(df)
        assert len(phi) == 3
        assert all(0 <= p <= 1 for p in phi)

    def test_combined_density_and_neutron(self):
        """Test composite from both density and neutron logs."""
        df = pd.DataFrame({
            'Bulk Density - Compensated kg/m3': [2200.0],  # ~0.276 phi
            'Neutron Porosity (Sandstone) Euc': [0.35]      # 0.35 phi
        })
        phi = compute_phi_combined(df)
        # Should be average
        expected = (0.276 + 0.35) / 2
        assert abs(phi[0] - expected) < 0.05

    def test_missing_both_logs(self):
        """Test with no relevant columns."""
        df = pd.DataFrame({'OTHER': [1.0, 2.0]})
        phi = compute_phi_combined(df)
        assert all(np.isnan(p) for p in phi)

    def test_nan_values_ignored(self):
        """Test that NaN values are ignored in composite."""
        df = pd.DataFrame({
            'Bulk Density - Compensated kg/m3': [2200.0, np.nan],
            'Neutron Porosity (Sandstone) Euc': [0.3, 0.35]
        })
        phi = compute_phi_combined(df)
        assert len(phi) == 2
        assert np.isfinite(phi[0])
        assert np.isfinite(phi[1])


class TestComputeFluidClass:
    """Test rule-based fluid classification."""

    def test_high_resistivity_reservoir(self):
        """Test high resistivity -> Potential Reservoir."""
        df = pd.DataFrame({
            'Resistivity Phase - Corrected - 2MHz ohm.m': [150.0],
            'Chrom 1 Total Gas Euc': [5.0],
            'PHI_COMBINED': [0.2]
        })
        fluid = compute_fluid_class(df)
        assert fluid[0] == 'Potential Reservoir'

    def test_high_gas_with_porosity(self):
        """Test high gas + porosity -> Potential Reservoir."""
        df = pd.DataFrame({
            'Resistivity Phase - Corrected - 2MHz ohm.m': [10.0],
            'Chrom 1 Total Gas Euc': [100.0],
            'PHI_COMBINED': [0.1]
        })
        fluid = compute_fluid_class(df)
        assert fluid[0] == 'Potential Reservoir'

    def test_moderate_resistivity_pay_zone(self):
        """Test moderate resistivity -> Pay Zone."""
        df = pd.DataFrame({
            'Resistivity Phase - Corrected - 2MHz ohm.m': [30.0],
            'Chrom 1 Total Gas Euc': [5.0],
            'PHI_COMBINED': [0.2]
        })
        fluid = compute_fluid_class(df)
        assert fluid[0] == 'Pay Zone'

    def test_moderate_gas_pay_zone(self):
        """Test moderate gas -> Pay Zone."""
        df = pd.DataFrame({
            'Resistivity Phase - Corrected - 2MHz ohm.m': [10.0],
            'Chrom 1 Total Gas Euc': [20.0],
            'PHI_COMBINED': [0.2]
        })
        fluid = compute_fluid_class(df)
        assert fluid[0] == 'Pay Zone'

    def test_background_low_values(self):
        """Test low values -> Background."""
        df = pd.DataFrame({
            'Resistivity Phase - Corrected - 2MHz ohm.m': [5.0],
            'Chrom 1 Total Gas Euc': [2.0],
            'PHI_COMBINED': [0.05]
        })
        fluid = compute_fluid_class(df)
        assert fluid[0] == 'Background'

    def test_missing_resistivity_and_gas(self):
        """Test with no resistivity/gas -> Background."""
        df = pd.DataFrame({'PHI_COMBINED': [0.2]})
        fluid = compute_fluid_class(df)
        assert fluid[0] == 'Background'

    def test_nan_handling(self):
        """Test NaN values handled gracefully."""
        df = pd.DataFrame({
            'Resistivity Phase - Corrected - 2MHz ohm.m': [np.nan, 30.0],
            'Chrom 1 Total Gas Euc': [5.0, np.nan],
            'PHI_COMBINED': [0.2, 0.2]
        })
        fluid = compute_fluid_class(df)
        assert len(fluid) == 2


class TestComputePorePressure:
    """Test simplified pore pressure estimation."""

    def test_hydrostatic_basic(self):
        """Test basic hydrostatic pressure calculation."""
        df = pd.DataFrame({
            'DEPTH': [1000.0],  # 1 km depth
            'Mud Weight In kg/m3': [1200.0],
            'Corrected Drilling Exponent unitless': [1.0]
        })
        pp = compute_pore_pressure(df)
        assert len(pp) == 1
        assert pp[0] > 0  # Pressure should be positive

    def test_depth_increases_pressure(self):
        """Test that deeper wells have higher pressure."""
        df = pd.DataFrame({
            'DEPTH': [1000.0, 2000.0, 3000.0],
            'Mud Weight In kg/m3': [1200.0, 1200.0, 1200.0],
            'Corrected Drilling Exponent unitless': [1.0, 1.0, 1.0]
        })
        pp = compute_pore_pressure(df)
        assert pp[0] < pp[1] < pp[2]

    def test_anomaly_from_low_exponent(self):
        """Test low drilling exponent increases pressure anomaly."""
        df_nominal = pd.DataFrame({
            'DEPTH': [2000.0],
            'Mud Weight In kg/m3': [1200.0],
            'Corrected Drilling Exponent unitless': [1.0]
        })
        df_low_exp = pd.DataFrame({
            'DEPTH': [2000.0],
            'Mud Weight In kg/m3': [1200.0],
            'Corrected Drilling Exponent unitless': [0.8]
        })
        pp_nominal = compute_pore_pressure(df_nominal)[0]
        pp_low = compute_pore_pressure(df_low_exp)[0]
        assert pp_low > pp_nominal  # Low exponent -> higher pressure

    def test_no_negative_pressure(self):
        """Test that pressure is never negative."""
        df = pd.DataFrame({
            'DEPTH': [100.0],
            'Mud Weight In kg/m3': [1000.0],
            'Corrected Drilling Exponent unitless': [2.0]  # High exponent (anomaly)
        })
        pp = compute_pore_pressure(df)
        assert pp[0] >= 0

    def test_missing_mud_weight_fallback(self):
        """Test fallback to water density when mud weight missing."""
        df = pd.DataFrame({
            'DEPTH': [1000.0],
            'Mud Weight In kg/m3': [np.nan],
            'Corrected Drilling Exponent unitless': [1.0]
        })
        pp = compute_pore_pressure(df)
        assert np.isfinite(pp[0])

    def test_missing_exponent_no_anomaly(self):
        """Test no anomaly when exponent is missing."""
        df = pd.DataFrame({
            'DEPTH': [1000.0],
            'Mud Weight In kg/m3': [1200.0],
            'Corrected Drilling Exponent unitless': [np.nan]
        })
        pp = compute_pore_pressure(df)
        assert np.isfinite(pp[0])


class TestComputeAllTargets:
    """Test all-targets computation."""

    def test_compute_all_targets_inplace(self):
        """Test that compute_all_targets modifies DataFrame in place."""
        df = pd.DataFrame({
            'DEPTH': [2000.0, 2100.0],
            'Bulk Density - Compensated kg/m3': [2200.0, 2250.0],
            'Neutron Porosity (Sandstone) Euc': [0.2, 0.25],
            'Resistivity Phase - Corrected - 2MHz ohm.m': [50.0, 60.0],
            'Chrom 1 Total Gas Euc': [10.0, 15.0],
            'Mud Weight In kg/m3': [1200.0, 1200.0],
            'Corrected Drilling Exponent unitless': [1.0, 1.0]
        })
        result = compute_all_targets(df, inplace=True)
        assert result is df  # Same object
        assert 'PHI_COMBINED' in df.columns
        assert 'FLUID_CLASS' in df.columns
        assert 'PREDICTED_PORE_PRESSURE_PSI' in df.columns

    def test_compute_all_targets_copy(self):
        """Test that compute_all_targets respects inplace=False."""
        df = pd.DataFrame({
            'DEPTH': [2000.0],
            'Bulk Density - Compensated kg/m3': [2200.0],
            'Neutron Porosity (Sandstone) Euc': [0.2],
            'Resistivity Phase - Corrected - 2MHz ohm.m': [50.0],
            'Chrom 1 Total Gas Euc': [10.0],
            'Mud Weight In kg/m3': [1200.0],
            'Corrected Drilling Exponent unitless': [1.0]
        })
        result = compute_all_targets(df, inplace=False)
        assert result is not df  # Different object
        assert 'PHI_COMBINED' not in df.columns  # Original unchanged
        assert 'PHI_COMBINED' in result.columns  # Copy has targets

    def test_skip_existing_targets(self):
        """Test that existing target columns are not recomputed."""
        df = pd.DataFrame({
            'DEPTH': [2000.0],
            'PHI_COMBINED': [0.3],  # Pre-existing
            'FLUID_CLASS': ['Pay Zone'],
            'PREDICTED_PORE_PRESSURE_PSI': [5000.0]
        })
        original_phi = df['PHI_COMBINED'].copy()
        compute_all_targets(df, inplace=True)
        assert (df['PHI_COMBINED'] == original_phi).all()

    def test_robust_to_missing_columns(self):
        """Test that function handles minimal input gracefully."""
        df = pd.DataFrame({
            'DEPTH': [2000.0]
            # All other columns missing
        })
        result = compute_all_targets(df, inplace=False)
        # Should have NaN for targets, not crash
        assert 'PHI_COMBINED' in result.columns
        assert 'FLUID_CLASS' in result.columns
        assert 'PREDICTED_PORE_PRESSURE_PSI' in result.columns
