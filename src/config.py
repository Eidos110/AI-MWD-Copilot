import os

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_PATH = os.path.join(DATA_DIR, "ready_modelling.csv")  # Added the missing constant

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
    'Pressure_Anomaly'
]

# Fallback: If any required feature is missing, use this minimal set
MINIMAL_FEATURES = ['DEPTH', 'Weight On Bit N', 'ROP for the Bit - Distance Over Time (On Bottom) m/s']

# Plotting configuration
PLOT_WIDTH = 18
PLOT_HEIGHT = 10
DEFAULT_SAMPLE_SIZE = 500
GR_MAX = 150
PORO_MAX = 0.4

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

# Default UI settings
DEFAULT_MIN_DEPTH = 2000
DEFAULT_MAX_DEPTH = 2500

# Fluid classification mapping
FLUID_CLASS_MAP = {'Background': 0, 'Pay Zone': 1, 'Potential Reservoir': 2}
FLUID_COLORS = ['lightgray', 'gold', 'lightcoral']

# Column names for plots (to avoid hardcoding)
COL_DEPTH = 'DEPTH'
COL_GAMMA_RAY = 'Gamma Ray - Corrected gAPI'
COL_RESISTIVITY = 'Resistivity Phase - Corrected - 2MHz ohm.m'
COL_POROSITY = 'PHI_COMBINED'
COL_FLUID_CLASS = 'FLUID_CLASS'
COL_PORE_PRESSURE = 'PREDICTED_PORE_PRESSURE_PSI'
COL_WOB = 'Weight On Bit N'
COL_TORQUE = 'Surface Torque Average N.m'

# Friendly names for SHAP interpretation (full-text readable labels)
SHAP_FEATURE_DISPLAY_NAMES = {
    'DEPTH': 'Well Depth (m)',
    'Gamma Ray - Corrected gAPI': 'Gamma Ray',
    'Resistivity Phase - Corrected - 2MHz ohm.m': 'Resistivity (Phase)',
    'Corrected Drilling Exponent unitless': 'Corrected Drilling Exponent',
    'ROP for the Bit - Distance Over Time (On Bottom) m/s': 'Rate of Penetration (ROP)',
    'Surface Torque Average N.m': 'Surface Torque',
    'Weight On Bit N': 'Weight on Bit (WOB)',
    'Chrom 1 Total Gas Euc': 'Chrome 1 Total Gas',
    'Mechanical Specific Energy Pa': 'Mechanical Specific Energy (MSE)',
    '28 Stick Slip RPM Average RPM': 'Stick Slip RPM',
    'Mud Weight In kg/m3': 'Mud Weight',
    'ECD at Bit kg/m3': 'Equivalent Circulating Density (ECD)',
    'Annular Pressure Pa': 'Annular Pressure',
    'DEPTH_FT': 'Depth (feet)',
    'P_Hydrostatic': 'Hydrostatic Pressure',
    'Delta_P_Hydro': 'Delta Hydrostatic Pressure',
    'P_Overburden': 'Overburden Pressure',
    'Effective_Stress': 'Effective Stress',
    'Pressure_Anomaly': 'Pressure Anomaly'
}