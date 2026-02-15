"""Target calculation helpers

Provides functions to compute:
- PHI_COMBINED: porosity composite (Wyllie density + neutron average)
- FLUID_CLASS: simple rule-based classification using resistivity, gas, porosity
- PREDICTED_PORE_PRESSURE_PSI: simplified Rehm & McClendon-style estimate

All functions accept a DataFrame and will not fail if optional columns
are missing; they leave existing target columns unchanged when present.
"""
from typing import Optional
import numpy as np
import pandas as pd


def _phi_from_density(rho_bulk, rho_matrix=2650.0, rho_fluid=1000.0):
    """Compute porosity from bulk density using Wyllie-style approximation.

    rho_bulk: bulk density in kg/m3
    rho_matrix: matrix density (default granite/sandstone ~2650 kg/m3)
    rho_fluid: fluid density (default water ~1000 kg/m3)
    Returns porosity as fraction (0-1). Values clipped to [0, 1].
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = (rho_matrix - rho_bulk) / (rho_matrix - rho_fluid)
    phi = np.asarray(phi, dtype=float)
    phi = np.where(np.isfinite(phi), phi, np.nan)
    return np.clip(phi, 0.0, 1.0)


def compute_phi_combined(df: pd.DataFrame, out_col: str = 'PHI_COMBINED') -> pd.Series:
    """Compute a composite porosity and return the Series.

    Strategy:
    - If bulk density available, compute Wyllie-like porosity.
    - If neutron porosity available, use it as-is.
    - Composite = mean of available porosity estimates (ignoring NaN).
    """
    bulk_col = 'Bulk Density - Compensated kg/m3'
    neutron_col = 'Neutron Porosity (Sandstone) Euc'

    n = len(df)
    phi_estimates = []

    if bulk_col in df.columns:
        phi_density = _phi_from_density(df[bulk_col].values)
        phi_estimates.append(phi_density)

    if neutron_col in df.columns:
        # Assume neutron log is already a fractional porosity or percent
        neutron = df[neutron_col].values
        # If values look like percent (>1), convert to fraction
        neutron = np.where(np.nan_to_num(neutron) > 1.5, neutron / 100.0, neutron)
        phi_estimates.append(neutron)

    if not phi_estimates:
        # Nothing to compute
        return pd.Series([np.nan] * n, index=df.index, name=out_col)

    stacked = np.vstack([np.asarray(a, dtype=float) for a in phi_estimates])
    # mean across available estimates, ignoring NaN
    phi_comb = np.nanmean(stacked, axis=0)
    phi_comb = np.clip(phi_comb, 0.0, 1.0)

    return pd.Series(phi_comb, index=df.index, name=out_col)


def compute_fluid_class(df: pd.DataFrame, phi_col: str = 'PHI_COMBINED', out_col: str = 'FLUID_CLASS') -> pd.Series:
    """Simple rule-based fluid classification.

    Rules (heuristic):
    - If resistivity or gas missing -> Background
    - If resistivity >= 100 OR (gas >= 50 and porosity >= 0.05) -> Potential Reservoir
    - Else if resistivity >= 20 OR gas >= 10 -> Pay Zone
    - Else -> Background
    """
    resist_col = 'Resistivity Phase - Corrected - 2MHz ohm.m'
    gas_col = 'Chrom 1 Total Gas Euc'

    n = len(df)
    result = pd.Series(['Background'] * n, index=df.index, name=out_col)

    if resist_col not in df.columns and gas_col not in df.columns:
        return result

    resist = df[resist_col] if resist_col in df.columns else pd.Series([np.nan] * n, index=df.index)
    gas = df[gas_col] if gas_col in df.columns else pd.Series([np.nan] * n, index=df.index)
    phi = df[phi_col] if phi_col in df.columns else pd.Series([np.nan] * n, index=df.index)

    # Potential Reservoir
    cond_pot = (resist.fillna(0) >= 100) | ((gas.fillna(0) >= 50) & (phi.fillna(0) >= 0.05))
    result.loc[cond_pot] = 'Potential Reservoir'

    # Pay Zone (exclude those already marked potential)
    cond_pay = ((resist.fillna(0) >= 20) | (gas.fillna(0) >= 10)) & (~cond_pot)
    result.loc[cond_pay] = 'Pay Zone'

    return result


def compute_pore_pressure(df: pd.DataFrame, out_col: str = 'PREDICTED_PORE_PRESSURE_PSI') -> pd.Series:
    """Compute a simplified pore pressure estimate (psi).

    Approach:
    - Compute hydrostatic pressure from mud weight if available: P = rho*g*depth
      convert Pa -> psi using 1 Pa = 0.0001450377 psi.
    - Use 'Corrected Drilling Exponent unitless' as an anomaly indicator; when
      exponent is lower than a nominal value (~1.0) we increase pore pressure.
    - This is a heuristic approximation of Rehm & McClendon behaviour for demo.
    """
    depth_col = 'DEPTH'
    mud_col = 'Mud Weight In kg/m3'
    dc_col = 'Corrected Drilling Exponent unitless'

    n = len(df)
    psi_result = pd.Series([np.nan] * n, index=df.index, name=out_col)

    # Compute hydrostatic psi
    if mud_col in df.columns and depth_col in df.columns:
        rho = df[mud_col].fillna(1000.0).values  # fallback to water
        depth = df[depth_col].fillna(0.0).values
        g = 9.80665
        # hydrostatic pressure in Pa
        p_hydro_pa = rho * g * depth
        pa_to_psi = 0.00014503773773020923
        p_hydro_psi = p_hydro_pa * pa_to_psi
    elif 'P_Hydrostatic' in df.columns:
        # If P_Hydrostatic exists assume it's in Pa
        p_hydro_psi = df['P_Hydrostatic'].fillna(0).values * 0.00014503773773020923
    else:
        p_hydro_psi = np.zeros(n)

    # Use corrected drilling exponent to compute anomaly
    if dc_col in df.columns:
        dc = df[dc_col].astype(float).values
        # nominal exponent (calibrated ~1.0). Positive anomaly when dc < nominal
        nominal = 1.0
        # scale factor tuned for demo -- 1000 psi per 0.1 deviation
        scale = 1000.0 / 0.1
        anomaly = (nominal - dc) * scale
    else:
        anomaly = np.zeros(n)

    psi = p_hydro_psi + anomaly
    # Ensure no negative pressures
    psi = np.where(np.isfinite(psi), psi, np.nan)
    psi = np.clip(psi, 0.0, None)

    psi_result[:] = psi
    return psi_result


def compute_all_targets(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """Compute all targets and return DataFrame (modifies inplace by default).

    Adds columns: `PHI_COMBINED`, `FLUID_CLASS`, `PREDICTED_PORE_PRESSURE_PSI`.
    """
    if not inplace:
        df = df.copy()

    # Porosity
    if 'PHI_COMBINED' not in df.columns:
        df['PHI_COMBINED'] = compute_phi_combined(df)

    # Fluid class
    if 'FLUID_CLASS' not in df.columns:
        df['FLUID_CLASS'] = compute_fluid_class(df, phi_col='PHI_COMBINED')

    # Pore pressure
    if 'PREDICTED_PORE_PRESSURE_PSI' not in df.columns:
        df['PREDICTED_PORE_PRESSURE_PSI'] = compute_pore_pressure(df)

    return df
