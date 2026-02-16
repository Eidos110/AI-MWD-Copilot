import joblib
import streamlit as st
import numpy as np
import pandas as pd
from src.config import (
    MODEL_POROSITY, MODEL_FLUID, MODEL_PRESSURE, ENCODER_FLUID,
    FEATURES_POROSITY, FEATURES_FLUID, FEATURES_PRESSURE, MINIMAL_FEATURES
)


class ModelManager:
    """
    Manages loading, validation, and inference for three predictive models.

    Models:
    - Porosity Model: Predicts PHI_COMBINED (composite porosity) using density/neutron features.
    - Fluid Model: Classifies fluid type (Background/Pay Zone/Potential Reservoir).
    - Pressure Model: Estimates pore pressure (PSI) using drilling dynamics + mud properties.

    Features are isolated by sensor physics to avoid data leakage and ensure robustness
    when sensors fail or data is missing.

    Attributes:
        porosity_model: Trained XGBoost regressor for porosity prediction.
        fluid_model: Trained XGBoost classifier for fluid classification.
        pressure_model: Trained XGBoost regressor for pore pressure estimation.
        fluid_encoder: LabelEncoder for fluid class labels.
    """
    def __init__(self):
        """Load all three models and the fluid label encoder from disk.
        
        Raises:
            FileNotFoundError: If any model file is missing.
            Exception: If loading fails for any reason.
        """
        try:
            self.porosity_model = joblib.load(MODEL_POROSITY)
            self.fluid_model = joblib.load(MODEL_FLUID)
            self.pressure_model = joblib.load(MODEL_PRESSURE)
            self.fluid_encoder = joblib.load(ENCODER_FLUID)
        except FileNotFoundError as e:
            st.error(f"Model file not found: {str(e)}")
            raise

    
    def predict_porosity(self, df):
        """Predict porosity values (PHI_COMBINED).
        
        Args:
            df (pd.DataFrame): Input features. Missing values are filled with column means.
        
        Returns:
            np.ndarray: Predicted porosity values (0-1 range).
        
        Raises:
            ValueError: If required features are missing and cannot fallback.
        """
        X = self._safe_select(df, FEATURES_POROSITY, "porosity")
        if X.empty:
            raise ValueError("No valid features available for porosity prediction")
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        return self.porosity_model.predict(X)
    
    def predict_fluid(self, df):
        """Predict fluid type and class probabilities.
        
        Fluid classes: 'Background', 'Pay Zone', 'Potential Reservoir'
        
        Args:
            df (pd.DataFrame): Input features (drilling dynamics + gamma ray).
        
        Returns:
            tuple: (fluid_classes, class_probabilities)
                - fluid_classes: np.ndarray of string labels.
                - class_probabilities: np.ndarray of shape (n_samples, n_classes).
        
        Raises:
            ValueError: If required features are missing.
        """
        X = self._safe_select(df, FEATURES_FLUID, "fluid")
        if X.empty:
            raise ValueError("No valid features available for fluid prediction")
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        pred_class = self.fluid_model.predict(X)
        pred_proba = self.fluid_model.predict_proba(X)
        return self.fluid_encoder.inverse_transform(pred_class), pred_proba
    
    def predict_pressure(self, df):
        """Predict pore pressure using the full FEATURES_PRESSURE list.

        If columns from the model's expected features are missing, they
        will be created and filled with sensible fallbacks so prediction
        can proceed. This ensures we use `FEATURES_PRESSURE` fully.
        """
        try:
            # Request full feature set (create missing cols as NaN)
            X = self._safe_select(df, FEATURES_PRESSURE, "pressure", force_full=True)
            if X.empty:
                raise ValueError("No valid features available for pressure prediction")

            # Fill NaNs with column mean where possible, then 0 as fallback
            X = X.fillna(X.mean()).fillna(0)

            # Get expected features from the model
            expected_features = self.pressure_model.get_booster().feature_names

            # If model expects features not present, add them as zeros
            missing_for_model = [f for f in expected_features if f not in X.columns]
            if missing_for_model:
                st.warning(f"⚠️ Pressure model expected extra features; adding missing: {missing_for_model}")
                for f in missing_for_model:
                    X[f] = 0

            # Reorder/select features in model's expected order
            X = X[expected_features]
            return self.pressure_model.predict(X)
        except Exception as e:
            st.warning(f"⚠️ Pressure prediction unavailable: {str(e)}")
            return None
    
    def _safe_select(self, df, features, model_name, force_full=False):
        """Select features with fallback and error handling.

        Strategy:
        1. If `force_full=True`: Return exactly the requested features (create NaN for missing).
        2. Otherwise: Select available features; if <3 available, fallback to MINIMAL_FEATURES.
        3. Warn if columns contain all-NaN values.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            features (list): Requested feature column names.
            model_name (str): Model name (for logging).
            force_full (bool): If True, return all features (fill missing with NaN).
        
        Returns:
            pd.DataFrame: Subset of df with selected features.
        
        Raises:
            ValueError: If no valid features remain after fallback.
        """
        if force_full:
            # Reindex to ensure we return exactly the requested features,
            # creating missing columns as NaN.
            selected = df.reindex(columns=features).copy()
            return selected

        available = [f for f in features if f in df.columns]

        if len(available) < 3:
            st.warning(f"Not enough features for {model_name} model. Using minimal set.")
            available = [f for f in MINIMAL_FEATURES if f in df.columns]

        if len(available) == 0:
            st.error(f"No valid features found for {model_name} prediction.")
            raise ValueError(f"Cannot proceed: no valid features for {model_name}")

        selected = df[available].copy()

        # Check for all NaN columns
        if selected.isna().all().any():
            problematic = selected.columns[selected.isna().all()].tolist()
            st.warning(f"Columns with all NaN values in {model_name}: {problematic}")

        return selected