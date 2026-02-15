# src/plots.py

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from src.config import (
    COL_DEPTH, COL_GAMMA_RAY, COL_RESISTIVITY, COL_POROSITY,
    COL_FLUID_CLASS, COL_PORE_PRESSURE, COL_WOB, COL_TORQUE,
    FLUID_CLASS_MAP, FLUID_COLORS, PLOT_WIDTH, PLOT_HEIGHT,
    GR_MAX, PORO_MAX
)

def plot_well_logs(df, predictions=None, depth_range=None):
    """Create multi-track well log plot"""
    if depth_range:
        mask = (df[COL_DEPTH] >= depth_range[0]) & (df[COL_DEPTH] <= depth_range[1])
        df = df[mask].copy()
    
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data in selected depth range", transform=ax.transAxes, ha='center')
        return fig

    fig, axes = plt.subplots(1, 6, figsize=(PLOT_WIDTH, PLOT_HEIGHT), sharey=True)
    fig.suptitle("AI-Powered MWD Copilot", fontsize=16, y=0.95)
    depth = df[COL_DEPTH]

    # Track 1: Gamma Ray
    if COL_GAMMA_RAY in df.columns:
        axes[0].plot(df[COL_GAMMA_RAY], depth, 'green', lw=1)
        axes[0].set_xlabel("GR (gAPI)")
    else:
        axes[0].text(0.5, 0.5, "No GR data", transform=axes[0].transAxes, ha='center', va='center')
        axes[0].set_xlabel("GR (gAPI)")
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, GR_MAX)

    # Track 2: Resistivity
    if COL_RESISTIVITY in df.columns:
        axes[1].semilogx(df[COL_RESISTIVITY], depth, 'red', lw=1)
        axes[1].set_xlabel("Resistivity (ohm.m)")
    else:
        axes[1].text(0.5, 0.5, "No Resistivity data", transform=axes[1].transAxes, ha='center', va='center')
        axes[1].set_xlabel("Resistivity (ohm.m)")
    axes[1].grid(True, alpha=0.3)

    # Track 3: Porosity
    axes[2].set_xlabel("Porosity")
    if COL_POROSITY in df.columns:
        axes[2].plot(df[COL_POROSITY], depth, 'blue', label='True', lw=1)
    if predictions and 'phi_pred' in predictions:
        axes[2].plot(predictions['phi_pred'], depth, 'orange', linestyle='--', label='Predicted', lw=1)
        axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, PORO_MAX)

    # Track 4: Fluid Type
    axes[3].set_xlabel("Fluid Type")
    axes[3].set_xticks([])
    if COL_FLUID_CLASS in df.columns:
        y_fluid = df[COL_FLUID_CLASS].map(FLUID_CLASS_MAP).fillna(0)
        scatter = axes[3].scatter([0]*len(y_fluid), depth, c=y_fluid, cmap=matplotlib.colors.ListedColormap(FLUID_COLORS), s=10)
        legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(FLUID_COLORS, FLUID_CLASS_MAP.keys())]
        axes[3].legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.8, 1))
    else:
        axes[3].text(0.5, 0.5, "No Fluid data", transform=axes[3].transAxes, ha='center', va='center')

    # Track 5: Pore Pressure
    axes[4].set_xlabel("Pore Press (psi)")
    if COL_PORE_PRESSURE in df.columns:
        axes[4].plot(df[COL_PORE_PRESSURE], depth, 'purple', lw=1)
    else:
        axes[4].text(0.5, 0.5, "No PP data", transform=axes[4].transAxes, ha='center', va='center')
    axes[4].grid(True, alpha=0.3)

    # Track 6: Drilling Parameters
    axes[5].set_xlabel("WOB (kN), Torque (kN.m)")
    has_wob = COL_WOB in df.columns
    has_torque = COL_TORQUE in df.columns
    
    if has_wob:
        axes[5].plot(df[COL_WOB]/1000, depth, 'black', label='WOB', lw=0.8)
    if has_torque:
        axes[5].plot(df[COL_TORQUE]/1000, depth, 'gray', label='Torque', lw=0.8)
    
    if has_wob or has_torque:
        axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    for ax in axes:
        if ax != axes[3]:  # Skip fluid track since it has special handling
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig