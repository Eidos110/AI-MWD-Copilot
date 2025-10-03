import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

MODEL_POROSITY = os.path.join(MODEL_DIR, "xgb_phi_model.pkl")
MODEL_FLUID = os.path.join(MODEL_DIR, "xgb_fluid_model.pkl")
MODEL_PRESSURE = os.path.join(MODEL_DIR, "xgb_pp_model_feat.pkl")
ENCODER_FLUID = os.path.join(MODEL_DIR, "le.pkl")

FEATURES_POROSITY = [
    'DEPTH',
    'Gamma Ray - Corrected gAPI',
    'Resistivity Phase - Corrected - 2MHz ohm.m',
    'Corrected Drilling Exponent unitless',
    'ROP for the Bit - Distance Over Time (On Bottom) m/s',
    'Surface Torque Average N.m',
    'Weight On Bit N',
    'Chrom 1 Total Gas Euc'
]

# Fluid Classification: Use drilling dynamics + gamma ray only
# Avoid direct resistivity/gas if simulating poor sensor conditions
FEATURES_FLUID = [
    'DEPTH',
    'Gamma Ray - Corrected gAPI',
    'Corrected Drilling Exponent unitless',
    'ROP for the Bit - Distance Over Time (On Bottom) m/s',
    'Mechanical Specific Energy Pa',  # add if available
    'Surface Torque Average N.m',
    'Weight On Bit N',
    '28 Stick Slip RPM Average RPM'
]

# Pore Pressure: Use mud props and drilling params
# Optional: include corrected exponent if available
FEATURES_PRESSURE = [
    'DEPTH',
    'Mud Weight In kg/m3',
    'ECD at Bit kg/m3',
    'Annular Pressure Pa',
    'ROP for the Bit - Distance Over Time (On Bottom) m/s',
    'Weight On Bit N',
    'Surface Torque Average N.m',
    'DEPTH_FT', 
    'P_Hydrostatic', 
    'Delta_P_Hydro', 
    'P_Overburden', 
    'Effective_Stress', 
    'Pressure_Anomaly',
]

# Fallback: If any required feature is missing, use this minimal set
MINIMAL_FEATURES = ['DEPTH', 'Weight On Bit N', 'ROP for the Bit - Distance Over Time (On Bottom) m/s']

# Display-only columns (for visualization)
DISPLAY_COLS = [
    'DEPTH',
    'Gamma Ray - Corrected gAPI',
    'Resistivity Phase - Corrected - 2MHz ohm.m',
    'Bulk Density - Compensated kg/m3',
    'Neutron Porosity (Sandstone) Euc',
    'PHI_COMBINED',
    'FLUID_CLASS',
    'PREDICTED_PORE_PRESSURE_PSI'
]

TARGETS = ['PHI_COMBINED','FLUID_CLASS','PREDICTED_PORE_PRESSURE_PSI']

DEFAULT_MIN_DEPTH = 2000
DEFAULT_MAX_DEPTH = 2500