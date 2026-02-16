# app.py
"""
AI-Powered MWD Copilot Dashboard

Real-time decision support for drilling operations using ML-predicted subsurface properties:
- Porosity (PHI_COMBINED) from density + neutron logs
- Fluid Type classification from drilling dynamics
- Pore Pressure from corrected drilling exponent

Features:
- Interactive depth range selection with presets
- SHAP model interpretability (3 tabs)
- Data quality reporting
- Prediction confidence scores & export
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import io

# Modules
from src.data_loader import load_data
from src.models import ModelManager
from src.plots import plot_well_logs
from src.shap_explainer import explain_model, get_shap_interpretation
from src.data_quality import generate_quality_report
from src.predictions import (compute_prediction_confidence, create_prediction_report,
                            export_predictions_csv, export_predictions_json)
from src.config import (DISPLAY_COLS, FEATURES_POROSITY, FEATURES_FLUID, FEATURES_PRESSURE, 
                       DEFAULT_SAMPLE_SIZE, SHAP_FEATURE_DISPLAY_NAMES)
from src.targets import compute_all_targets


# Page setup
st.set_page_config(page_title="AI MWD Copilot", layout="wide")
st.title("🛰️ AI-Powered MWD Copilot Dashboard")

@st.cache_resource
def get_models():
    return ModelManager()

@st.cache_data
def get_data():
    return load_data()

# --- Sidebar file uploader is placed before loading data so uploaded files are preferred
# Users can upload CSV or Excel files which must include a 'DEPTH' column
st.sidebar.header("🔍 View Settings")
st.sidebar.subheader("📁 Upload Your Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file (must include 'DEPTH' column)",
    type=["csv", "xlsx"],
    help="If you upload a file, it will replace the default dataset for this session."
)
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            user_df = pd.read_csv(uploaded_file)
        else:
            user_df = pd.read_excel(uploaded_file)

        if 'DEPTH' not in user_df.columns:
            st.sidebar.error("Uploaded file must contain a 'DEPTH' column.")
        else:
            user_df = user_df.sort_values('DEPTH').reset_index(drop=True)
            try:
                # Compute derived targets defensively; do not modify original uploaded object
                user_df = compute_all_targets(user_df, inplace=False)
            except Exception:
                st.sidebar.warning("Could not compute derived targets automatically; proceeding with uploaded data.")

            st.sidebar.success(f"Uploaded dataset loaded ({len(user_df)} rows).")
            st.session_state['uploaded_df'] = user_df
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {str(e)}")

# Load data (prefer uploaded dataset stored in session)
try:
    df = get_data()
    if st.session_state.get('uploaded_df') is not None:
        df = st.session_state.get('uploaded_df')
    models = get_models()
    st.success(f"✅ Loaded {len(df)} depth points.")
except Exception as e:
    st.error(f"❌ Failed to load data/models: {str(e)}")
    st.stop()

# Depth range presets
st.sidebar.subheader("📍 Depth Range Presets")
preset_buttons = st.sidebar.columns(3)
with preset_buttons[0]:
    if st.button("Shallow (2000-2500m)"):
        st.session_state.min_depth = 2000
        st.session_state.max_depth = 2500
with preset_buttons[1]:
    if st.button("Mid (2500-3000m)"):
        st.session_state.min_depth = 2500
        st.session_state.max_depth = 3000
with preset_buttons[2]:
    if st.button("Deep (3000-3500m)"):
        st.session_state.min_depth = 3000
        st.session_state.max_depth = 3500

# Custom depth range
st.sidebar.subheader("Custom Range")
min_depth = st.sidebar.number_input(
    "Min Depth (m)", 
    int(df['DEPTH'].min()), 
    int(df['DEPTH'].max()), 
    st.session_state.get('min_depth', 2000)
)
max_depth = st.sidebar.number_input(
    "Max Depth (m)", 
    int(df['DEPTH'].min()), 
    int(df['DEPTH'].max()), 
    st.session_state.get('max_depth', 2500)
)

# Store in session
st.session_state.min_depth = min_depth
st.session_state.max_depth = max_depth

show_predictions = st.sidebar.checkbox("Show Predictions", True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
show_data_quality = st.sidebar.checkbox("Show Data Quality Report", False)

# Validate depth range
if min_depth >= max_depth:
    st.error("❌ Min Depth must be less than Max Depth!")
    st.stop()

# Filter data
mask = (df['DEPTH'] >= min_depth) & (df['DEPTH'] <= max_depth)
df_view = df[mask].copy()

if len(df_view) == 0:
    st.warning("No data in selected depth range.")
else:
    predictions = {}
    confidences = {}
    
    if show_predictions:
        with st.spinner("Running AI models..."):
            try:
                # Only make predictions if the models have been loaded successfully
                if hasattr(models, 'porosity_model'):
                    predictions['phi_pred'] = models.predict_porosity(df_view)
                    if show_confidence:
                        confidences['phi_conf'] = compute_prediction_confidence(
                            models.porosity_model, 
                            models._safe_select(df_view, FEATURES_POROSITY, "porosity"),
                            predictions['phi_pred']
                        )
                
                if hasattr(models, 'fluid_model') and hasattr(models, 'fluid_encoder'):
                    fluid_preds, fluid_probs = models.predict_fluid(df_view)
                    predictions['fluid_pred'] = fluid_preds
                    predictions['fluid_prob'] = fluid_probs
                    if show_confidence:
                        confidences['fluid_conf'] = fluid_probs.max(axis=1)
                
                if hasattr(models, 'pressure_model'):
                    pp_pred = models.predict_pressure(df_view)
                    if pp_pred is not None:
                        predictions['pp_pred'] = pp_pred
                        if show_confidence:
                            confidences['pp_conf'] = compute_prediction_confidence(
                                models.pressure_model,
                                models._safe_select(df_view, FEATURES_PRESSURE, "pressure", force_full=True),
                                pp_pred
                            )
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    # Data Quality Dashboard
    if show_data_quality:
        with st.expander("📊 Data Quality Report", expanded=False):
            quality_report = generate_quality_report(
                df_view,
                critical_cols=DISPLAY_COLS,
                feature_groups={
                    'Porosity': FEATURES_POROSITY,
                    'Fluid': FEATURES_FLUID,
                    'Pressure': FEATURES_PRESSURE
                }
            )
            st.markdown(quality_report['summary'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Missing Values by Column")
                st.dataframe(quality_report['missing_values'].head(10), use_container_width=True)
            with col2:
                st.subheader("Sensor Health Scores")
                st.dataframe(quality_report['sensor_health'], use_container_width=True)

    # Plot
    fig = plot_well_logs(df_view, predictions, depth_range=[min_depth, max_depth])
    st.pyplot(fig)

    # SHAP - Model Interpretability with Tabs
    st.header("Model Interpretability")
    
    # Create tabs for each model explanation
    shap_tabs = st.tabs(["💧 Porosity", "💧 Fluid Type", "⚡ Pore Pressure"])
    
    # Tab 1: Porosity Model
    with shap_tabs[0]:
        if st.button("Explain Porosity Model", key="porosity_shap"):
            with st.spinner("Computing SHAP values for Porosity..."):
                try:
                    sample_df = models._safe_select(df_view, FEATURES_POROSITY, "porosity")
                    if len(sample_df) == 0:
                        st.warning("Not enough valid features for Porosity model.")
                    else:
                        sample = sample_df.sample(n=min(DEFAULT_SAMPLE_SIZE, len(sample_df)), random_state=42)
                        
                        # Get interpretation
                        interpretation = get_shap_interpretation(
                            models.porosity_model,
                            sample,
                            feature_names=sample.columns.tolist(),
                            display_names=SHAP_FEATURE_DISPLAY_NAMES,
                            top_n=5
                        )
                        
                        st.markdown(interpretation['summary'])
                        
                        shap_fig = explain_model(
                            models.porosity_model,
                            sample,
                            feature_names=[c.split(' ')[0] for c in sample.columns],
                            title="SHAP: Porosity Prediction Drivers"
                        )
                        st.pyplot(shap_fig)
                        
                        if interpretation['data'] is not None:
                            st.subheader("📈 Feature Importance Ranking")
                            st.dataframe(interpretation['data'], use_container_width=True)
                
                except Exception as e:
                    st.error(f"SHAP analysis failed: {str(e)}")
    
    # Tab 2: Fluid Type Model
    with shap_tabs[1]:
        if st.button("Explain Fluid Type Model", key="fluid_shap"):
            with st.spinner("Computing SHAP values for Fluid Type..."):
                try:
                    sample_df = models._safe_select(df_view, FEATURES_FLUID, "fluid")
                    if len(sample_df) == 0:
                        st.warning("Not enough valid features for Fluid Type model.")
                    else:
                        sample = sample_df.sample(n=min(DEFAULT_SAMPLE_SIZE, len(sample_df)), random_state=42)
                        
                        # Get interpretation
                        interpretation = get_shap_interpretation(
                            models.fluid_model,
                            sample,
                            feature_names=sample.columns.tolist(),
                            display_names=SHAP_FEATURE_DISPLAY_NAMES,
                            top_n=5
                        )
                        
                        st.markdown(interpretation['summary'])
                        
                        shap_fig = explain_model(
                            models.fluid_model,
                            sample,
                            feature_names=[c.split(' ')[0] for c in sample.columns],
                            title="SHAP: Fluid Type Prediction Drivers"
                        )
                        st.pyplot(shap_fig)
                        
                        if interpretation['data'] is not None:
                            st.subheader("📈 Feature Importance Ranking")
                            st.dataframe(interpretation['data'], use_container_width=True)
                
                except Exception as e:
                    st.error(f"SHAP analysis failed: {str(e)}")
    
    # Tab 3: Pore Pressure Model
    with shap_tabs[2]:
        if st.button("Explain Pore Pressure Model", key="pressure_shap"):
            with st.spinner("Computing SHAP values for Pore Pressure..."):
                try:
                    sample_df = models._safe_select(df_view, FEATURES_PRESSURE, "pressure", force_full=True)
                    if len(sample_df) == 0:
                        st.warning("Not enough valid features for Pore Pressure model.")
                    else:
                        sample = sample_df.sample(n=min(DEFAULT_SAMPLE_SIZE, len(sample_df)), random_state=42)
                        
                        # Get interpretation
                        interpretation = get_shap_interpretation(
                            models.pressure_model,
                            sample,
                            feature_names=sample.columns.tolist(),
                            display_names=SHAP_FEATURE_DISPLAY_NAMES,
                            top_n=5
                        )
                        
                        st.markdown(interpretation['summary'])
                        
                        shap_fig = explain_model(
                            models.pressure_model,
                            sample,
                            feature_names=[c.split(' ')[0] for c in sample.columns],
                            title="SHAP: Pore Pressure Prediction Drivers"
                        )
                        st.pyplot(shap_fig)
                        
                        if interpretation['data'] is not None:
                            st.subheader("📈 Feature Importance Ranking")
                            st.dataframe(interpretation['data'], use_container_width=True)
                
                except Exception as e:
                    st.error(f"SHAP analysis failed: {str(e)}")

    # Data sample & Export Section
    st.header("📊 Data & Export")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Raw Data Sample")
        st.dataframe(df_view[DISPLAY_COLS].head(10), use_container_width=True)
    
    with col2:
        st.subheader("Export Options")
        if st.button("📥 Export as CSV"):
            csv_path = export_predictions_csv(df_view, predictions, 'predictions.csv')
            with open(csv_path, 'rb') as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )
        
        if st.button("📤 Export as JSON"):
            json_path = export_predictions_json(
                df_view, 
                predictions,
                metadata={'model_version': '1.0', 'export_date': datetime.now().isoformat()},
                output_path='predictions.json'
            )
            with open(json_path, 'r') as f:
                st.download_button(
                    label="Download JSON",
                    data=f,
                    file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                    mime='application/json'
                )
    
    # Prediction Report with Confidence
    if show_confidence and predictions:
        st.subheader("🎯 Prediction Report with Confidence Intervals")
        report_df = create_prediction_report(df_view, predictions, confidences)
        st.dataframe(report_df, use_container_width=True)
        
        # Download report
        if st.button("📊 Download Prediction Report"):
            report_path = 'prediction_report.csv'
            report_df.to_csv(report_path, index=False)
            with open(report_path, 'rb') as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name=f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )

# Footer
st.markdown("---")
st.markdown("🔬 AI-Powered MWD Copilot | Built with Happy for intelligent drilling | Portfolio Project 2025")