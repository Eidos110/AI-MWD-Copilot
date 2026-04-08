import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd

def explain_model(model, X_sample, feature_names, title="SHAP Summary"):
    """Generate SHAP explanation plot for model predictions.
    
    This function gracefully handles errors during SHAP computation and
    returns a fallback plot on failure (e.g., memory limits, unsupported model types).
    
    Args:
        model: Trained model (XGBoost, sklearn, or compatible).
        X_sample (pd.DataFrame): Feature matrix for explanation.
        feature_names (list): Feature column names (for plot labels).
        title (str): Plot title.
    
    Returns:
        matplotlib.figure.Figure: SHAP plot (or fallback error message).
    """
    try:
        # Ensure feature names match sample columns if not provided
        if feature_names is None or len(feature_names) != X_sample.shape[1]:
            feature_names = X_sample.columns.tolist() if hasattr(X_sample, 'columns') else [f"Feature_{i}" for i in range(X_sample.shape[1])]
        
        # Check model type to select appropriate explainer
        if isinstance(model, xgb.XGBModel):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # For non-tree models, use the generic explainer
            explainer = shap.Explainer(model.predict, X_sample)
            shap_values = explainer(X_sample)
        
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, shap.Explanation):
            # For newer SHAP API
            shap.plots.beeswarm(shap_values, show=False)
        else:
            # For older SHAP API with numerical shap_values
            # Handle multi-class classification case
            if isinstance(shap_values, list):
                # Multi-class: average across classes for visualization
                shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
            elif shap_values.ndim == 3:
                # Shape: (n_samples, n_features, n_classes)
                shap_values = np.mean(np.abs(shap_values), axis=2)
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    except Exception as e:
        # Fallback: show error message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"SHAP analysis failed: {str(e)}", 
                transform=ax.transAxes, ha='center', va='center', wrap=True)
        ax.set_title(title)
        return fig


def get_shap_interpretation(model, X_sample, feature_names=None, top_n=5, display_names=None):
    """Generate human-readable SHAP interpretation (feature importance + impact direction).
    
    This function computes SHAP values and produces:
    1. Textual summary (top features pushing predictions UP/DOWN)
    2. Feature importance ranking (mean absolute SHAP values)
    3. Data DataFrame for audit/visualization
    
    Gracefully handles missing columns and computation errors.
    
    Args:
        model: Trained model (XGBoost, sklearn, or compatible).
        X_sample (pd.DataFrame or np.ndarray): Feature matrix.
        feature_names (list, optional): Original column names. Defaults to X_sample.columns.
        top_n (int): Number of top drivers to display. Defaults to 5.
        display_names (dict, optional): Mapping from original names to user-friendly names.
    
    Returns:
        dict with keys:
            'summary' (str): Markdown-formatted interpretation.
            'top_positive' (list): Top features with positive impact (as dicts).
            'top_negative' (list): Top features with negative impact (as dicts).
            'data' (pd.DataFrame): Full ranking table.
    """
    try:
        # Get feature names
        if feature_names is None or len(feature_names) != X_sample.shape[1]:
            feature_names = X_sample.columns.tolist() if hasattr(X_sample, 'columns') else [f"Feature_{i}" for i in range(X_sample.shape[1])]
        
        # Create display names mapping if not provided
        if display_names is None:
            display_names = {name: name for name in feature_names}
        
        # Calculate SHAP values
        if isinstance(model, xgb.XGBModel):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model.predict, X_sample)
            shap_values = explainer(X_sample)
        
        # Handle multi-class case: aggregate across classes
        if isinstance(shap_values, list):
            # Multi-class: list of arrays, one per class
            # Stack and take mean absolute values across classes
            shap_values = np.array(shap_values)  # (n_classes, n_samples, n_features)
            shap_values = np.mean(np.abs(shap_values), axis=0)  # Average across classes -> (n_samples, n_features)
        elif isinstance(shap_values, shap.Explanation):
            # Newer SHAP API
            shap_values = shap_values.values
        elif shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) - aggregate across classes
            shap_values = np.mean(np.abs(shap_values), axis=2)  # -> (n_samples, n_features)
        
        # Ensure shap_values is 2D (n_samples, n_features)
        if shap_values.ndim != 2:
            raise ValueError(f"SHAP values have unexpected shape: {shap_values.shape}")
        
        # Calculate mean absolute SHAP values (global feature importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create ranking DataFrame with both original and display names
        importance_df = pd.DataFrame({
            'Original Name': feature_names,
            'Display Name': [display_names.get(f, f) for f in feature_names],
            'Mean |SHAP|': mean_abs_shap,
            'Direction': ['↑ PUSH UP' if np.mean(shap_values[:, i]) > 0 else '↓ PUSH DOWN' 
                         for i in range(len(feature_names))]
        }).sort_values('Mean |SHAP|', ascending=False)
        
        # Get top positive and negative contributors
        top_positive = importance_df[importance_df['Direction'] == '↑ PUSH UP'].head(top_n)
        top_negative = importance_df[importance_df['Direction'] == '↓ PUSH DOWN'].head(top_n)
        
        # Build interpretation text using display names
        summary = "### 🔍 SHAP Model Interpretation Summary\n\n"
        summary += f"**Analysis of {len(X_sample)} samples**\n\n"
        
        summary += "#### ⬆️ Top Features Increasing Predictions (Positive Impact):\n"
        if not top_positive.empty:
            for idx, row in top_positive.iterrows():
                summary += f"- **{row['Display Name']}**: {row['Mean |SHAP|']:.4f} (avg contribution)\n"
        else:
            summary += "- No strongly positive features detected\n"
        
        summary += "\n#### ⬇️ Top Features Decreasing Predictions (Negative Impact):\n"
        if not top_negative.empty:
            for idx, row in top_negative.iterrows():
                summary += f"- **{row['Display Name']}**: {row['Mean |SHAP|']:.4f} (avg contribution)\n"
        else:
            summary += "- No strongly negative features detected\n"
        
        summary += f"\n#### 📊 Overall Insight:\n"
        summary += f"The model's predictions are primarily driven by {top_n} key features. "
        summary += f"The **{importance_df.iloc[0]['Display Name']}** has the strongest influence with "
        summary += f"an average |SHAP| value of **{importance_df.iloc[0]['Mean |SHAP|']:.4f}**.\n"
        
        return {
            'summary': summary,
            'top_positive': top_positive.to_dict('records'),
            'top_negative': top_negative.to_dict('records'),
            'data': importance_df
        }
    
    except Exception as e:
        return {
            'summary': f"❌ Interpretation failed: {str(e)}\n\n**Debugging tips:**\n- Check feature availability in your data\n- Ensure model is compatible with SHAP (XGBoost, sklearn)\n- Try with a smaller sample size if memory-limited",
            'top_positive': [],
            'top_negative': [],
            'data': None
        }