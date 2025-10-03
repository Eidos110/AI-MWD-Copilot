# app.py

import streamlit as st
import matplotlib.pyplot as plt

# Modules
from src.data_loader import load_data
from src.models import ModelManager
from src.plots import plot_well_logs
from src.shap_explainer import explain_model
from src.config import DISPLAY_COLS, FEATURES_POROSITY


# Page setup
st.set_page_config(page_title="AI MWD Copilot", layout="wide")
st.title("🛰️ AI-Powered MWD Copilot Dashboard")

@st.cache_resource
def get_models():
    return ModelManager()

@st.cache_data
def get_data():
    return load_data()

# Load data
try:
    df = get_data()
    models = get_models()
    st.success(f"✅ Loaded {len(df)} depth points.")
except Exception as e:
    st.error(f"❌ Failed to load data/models: {str(e)}")
    st.stop()

# Sidebar
st.sidebar.header("🔍 View Settings")
min_depth = st.sidebar.number_input("Min Depth", int(df['DEPTH'].min()), int(df['DEPTH'].max()), 2000)
max_depth = st.sidebar.number_input("Max Depth", int(df['DEPTH'].min()), int(df['DEPTH'].max()), 2500)
show_predictions = st.sidebar.checkbox("Show Predictions", True)

# Filter data
mask = (df['DEPTH'] >= min_depth) & (df['DEPTH'] <= max_depth)
df_view = df[mask].copy()

if len(df_view) == 0:
    st.warning("No data in selected depth range.")
else:
    predictions = {}
    if show_predictions:
        with st.spinner("Running AI models..."):
            try:
                predictions['phi_pred'] = models.predict_porosity(df_view)
                predictions['fluid_pred'], _ = models.predict_fluid(df_view)
                predictions['pp_pred'] = models.predict_pressure(df_view)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    # Plot
    fig = plot_well_logs(df_view, predictions, depth_range=[min_depth, max_depth])
    st.pyplot(fig)

    # SHAP
    st.header("🧠 Model Interpretability")
    if st.button("Explain Porosity Model"):
        with st.spinner("Computing SHAP values..."):
            sample = df_view[models._safe_select(df_view, FEATURES_POROSITY, "porosity")].sample(n=min(500, len(df_view)), random_state=42)
            shap_fig = explain_model(
                models.porosity_model,
                sample,
                feature_names=[c.split(' ')[0] for c in sample.columns],
                title="SHAP: Porosity Prediction Drivers"
            )
            st.pyplot(shap_fig)

    # Data sample
    st.header("📊 Raw Data Sample")
    st.dataframe(df_view[DISPLAY_COLS].head(10))

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for intelligent drilling | Portfolio Project 2025")