# src/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_well_logs(df, predictions=None, depth_range=None):
    """Create multi-track well log plot"""
    if depth_range:
        mask = (df['DEPTH'] >= depth_range[0]) & (df['DEPTH'] <= depth_range[1])
        df = df[mask].copy()
    
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data in selected depth range", transform=ax.transAxes, ha='center')
        return fig

    fig, axes = plt.subplots(1, 6, figsize=(18, 10), sharey=True)
    fig.suptitle("AI-Powered MWD Copilot", fontsize=16, y=0.95)
    depth = df['DEPTH']

    # Track 1: Gamma Ray
    axes[0].plot(df['Gamma Ray - Corrected gAPI'], depth, 'green', lw=1)
    axes[0].set_xlabel("GR (gAPI)")
    axes[0].invert_yaxis(); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 150)

    # Track 2: Resistivity
    axes[1].semilogx(df['Resistivity Phase - Corrected - 2MHz ohm.m'], depth, 'red', lw=1)
    axes[1].set_xlabel("Resistivity (ohm.m)")
    axes[1].grid(True, alpha=0.3)

    # Track 3: Porosity
    axes[2].plot(df['PHI_COMBINED'], depth, 'blue', label='True', lw=1)
    if predictions and 'phi_pred' in predictions:
        axes[2].plot(predictions['phi_pred'], depth, 'orange', linestyle='--', label='Predicted', lw=1)
        axes[2].legend()
    axes[2].set_xlabel("Porosity")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 0.4)

    # Track 4: Fluid Type
    fluid_map = {'Background': 0, 'Pay Zone': 1, 'Potential Reservoir': 2}
    y_fluid = df['FLUID_CLASS'].map(fluid_map).fillna(0)
    colors = ['lightgray', 'gold', 'lightcoral']
    scatter = axes[3].scatter([0]*len(y_fluid), depth, c=y_fluid, cmap=plt.cm.colors.ListedColormap(colors), s=10)
    axes[3].set_xlabel("Fluid Type")
    axes[3].set_xticks([])

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, fluid_map.keys())]
    axes[3].legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.8, 1))

    # Track 5: Pore Pressure
    axes[4].plot(df['PREDICTED_PORE_PRESSURE_PSI'], depth, 'purple', lw=1)
    axes[4].set_xlabel("Pore Press (psi)")
    axes[4].grid(True, alpha=0.3)

    # Track 6: Drilling Parameters
    axes[5].plot(df['Weight On Bit N']/1000, depth, 'black', label='WOB', lw=0.8)
    axes[5].plot(df['Surface Torque Average N.m']/1000, depth, 'gray', label='Torque', lw=0.8)
    axes[5].set_xlabel("WOB (kN), Torque (kN.m)")
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    for ax in axes:
        if ax != axes[3]:
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig