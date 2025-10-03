import joblib
import streamlit as st
import numpy as np
from src.config import (
    MODEL_POROSITY, MODEL_FLUID, MODEL_PRESSURE, ENCODER_FLUID,
    FEATURES_POROSITY, FEATURES_FLUID, FEATURES_PRESSURE, MINIMAL_FEATURES
)


class ModelManager:
    def __init__(self):
        self.porosity_model = joblib.load(MODEL_POROSITY)
        self.fluid_model = joblib.load(MODEL_FLUID)
        self.pressure_model = joblib.load(MODEL_PRESSURE)
        self.fluid_encoder = joblib.load(ENCODER_FLUID)
    
    def predict_porosity(self, df):
        X = self._safe_select(df, FEATURES_POROSITY, "porosity")
        return self.porosity_model.predict(X)
    
    def predict_fluid(self, df):
        X = self._safe_select(df, FEATURES_FLUID, "fluid")
        pred_class = self.fluid_model.predict(X)
        pred_proba = self.fluid_model.predict_proba(X)
        return self.fluid_encoder.inverse_transform(pred_class), pred_proba
    
    def predict_pressure(self, df):
        X = self._safe_select(df, FEATURES_PRESSURE, "pressure")
        return self.pressure_model.predict(X)
    
    def _safe_select(self, df, features, model_name):
        """Select features with fallback and error handling"""
        available = [f for f in features if f in df.columns]
        
        if len(available) < 3:
            st.warning(f"Not enough features for {model_name} model. Using minimal set.")
            available = [f for f in MINIMAL_FEATURES if f in df.columns]
        
        if len(available) == 0:
            raise ValueError(f"No valid features found for {model_name} prediction.")
            
        return df[available].copy()