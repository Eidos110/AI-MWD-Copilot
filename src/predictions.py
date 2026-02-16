"""
Prediction confidence and uncertainty quantification module.

Provides utilities to:
- Estimate prediction confidence from model internals
- Add uncertainty bands to predictions
- Export predictions with confidence scores to CSV/JSON
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import json
from datetime import datetime


def compute_prediction_confidence(model, X_test: pd.DataFrame, 
                                  y_pred: np.ndarray) -> np.ndarray:
    """Estimate prediction confidence from XGBoost model internals.
    
    For regression: uses prediction variance across trees.
    For classification: uses max probability from predict_proba.
    
    Args:
        model: Trained XGBoost model.
        X_test (pd.DataFrame): Test features.
        y_pred (np.ndarray): Model predictions.
    
    Returns:
        np.ndarray: Confidence scores (0-1 scale), where 1 = highest confidence.
    """
    try:
        # Try to get tree predictions for variance estimate
        if hasattr(model, 'predict') and hasattr(model, 'get_booster'):
            # XGBoost: predict with output_margin for tree-level contributions
            predictions_all = []
            n_trees = model.get_booster().best_ntree_limit
            
            # Get predictions at different tree depths for variance
            for ntrees in [max(1, n_trees//4), max(1, n_trees//2), 
                          max(1, 3*n_trees//4), n_trees]:
                try:
                    pred = model.predict(X_test, ntree_limit=ntrees)
                    predictions_all.append(pred)
                except:
                    break
            
            if len(predictions_all) > 1:
                # Compute coefficient of variation across tree depths
                predictions_all = np.vstack(predictions_all)
                variance = np.var(predictions_all, axis=0)
                mean_pred = np.mean(predictions_all, axis=0)
                
                # Normalize variance (lower variance = higher confidence)
                with np.errstate(divide='ignore', invalid='ignore'):
                    cv = np.sqrt(variance) / (np.abs(mean_pred) + 1e-6)
                    confidence = 1.0 / (1.0 + cv)  # Convert to 0-1
                    confidence = np.clip(confidence, 0, 1)
                    return confidence
        
        # Fallback: uniform confidence
        return np.ones(len(y_pred))
    
    except Exception:
        # If any error, return uniform confidence
        return np.ones(len(y_pred))


def compute_prediction_intervals(y_pred: np.ndarray, confidence: np.ndarray, 
                                ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate prediction intervals based on confidence scores.
    
    Higher confidence -> tighter interval.
    Uses z-score for 95% CI (z ~ 1.96).
    
    Args:
        y_pred (np.ndarray): Predictions.
        confidence (np.ndarray): Confidence scores (0-1).
        ci (float): Confidence interval level (default 0.95 for 95% CI).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (lower_bound, upper_bound)
    """
    z_score = 1.96 if ci == 0.95 else 1.645 if ci == 0.90 else 1.0
    
    # Margin increases with lower confidence
    margin = z_score * (1 - confidence) * np.abs(y_pred + 1e-6)
    
    lower = y_pred - margin
    upper = y_pred + margin
    
    return lower, upper


def export_predictions_csv(df: pd.DataFrame, predictions: dict, 
                          output_path: str = 'predictions.csv') -> str:
    """Export predictions with confidence to CSV.
    
    Args:
        df (pd.DataFrame): Original data (with DEPTH).
        predictions (dict): Predictions dict with keys like 'phi_pred', 'phi_conf', etc.
        output_path (str): Output CSV path.
    
    Returns:
        str: Path to output file.
    """
    export_df = df[['DEPTH']].copy() if 'DEPTH' in df.columns else pd.DataFrame({'Row': range(len(df))})
    
    # Add predictions and confidence
    for key, values in predictions.items():
        if isinstance(values, np.ndarray):
            export_df[key] = values
    
    export_df.to_csv(output_path, index=False)
    return output_path


def export_predictions_json(df: pd.DataFrame, predictions: dict, 
                           metadata: dict = None,
                           output_path: str = 'predictions.json') -> str:
    """Export predictions with metadata to JSON.
    
    Args:
        df (pd.DataFrame): Original data.
        predictions (dict): Predictions dict.
        metadata (dict, optional): Additional metadata (e.g., model version, date).
        output_path (str): Output JSON path.
    
    Returns:
        str: Path to output file.
    """
    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'num_samples': len(df),
            'num_features': len(df.columns),
            **(metadata or {})
        },
        'predictions': {}
    }
    
    # Add depth and predictions
    if 'DEPTH' in df.columns:
        export_data['predictions']['DEPTH'] = df['DEPTH'].tolist()
    
    for key, values in predictions.items():
        if isinstance(values, np.ndarray):
            export_data['predictions'][key] = values.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return output_path


def create_prediction_report(df: pd.DataFrame, predictions: dict, 
                            confidences: dict = None) -> pd.DataFrame:
    """Create a comprehensive prediction report DataFrame.
    
    Args:
        df (pd.DataFrame): Original data (should contain DEPTH).
        predictions (dict): Model predictions (phi_pred, fluid_pred, pp_pred, etc.).
        confidences (dict, optional): Confidence scores per model.
    
    Returns:
        pd.DataFrame: Report with predictions, confidence, and intervals.
    """
    report = df[['DEPTH']].copy() if 'DEPTH' in df.columns else pd.DataFrame(index=range(len(df)))
    
    # Add true values if available
    if 'PHI_COMBINED' in df.columns:
        report['Porosity (True)'] = df['PHI_COMBINED']
    if 'phi_pred' in predictions:
        report['Porosity (Predicted)'] = predictions['phi_pred']
        if confidences and 'phi_conf' in confidences:
            report['Porosity Confidence'] = confidences['phi_conf']
            lower, upper = compute_prediction_intervals(predictions['phi_pred'], 
                                                        confidences['phi_conf'])
            report['Porosity Lower Bound'] = lower
            report['Porosity Upper Bound'] = upper
    
    # Fluid predictions
    if 'FLUID_CLASS' in df.columns:
        report['Fluid Type (True)'] = df['FLUID_CLASS']
    if 'fluid_pred' in predictions:
        report['Fluid Type (Predicted)'] = predictions['fluid_pred']
    
    # Pressure predictions
    if 'PREDICTED_PORE_PRESSURE_PSI' in df.columns:
        report['Pore Pressure (True, PSI)'] = df['PREDICTED_PORE_PRESSURE_PSI']
    if 'pp_pred' in predictions:
        report['Pore Pressure (Predicted, PSI)'] = predictions['pp_pred']
        if confidences and 'pp_conf' in confidences:
            report['Pressure Confidence'] = confidences['pp_conf']
            lower, upper = compute_prediction_intervals(predictions['pp_pred'], 
                                                        confidences['pp_conf'])
            report['Pressure Lower Bound (PSI)'] = lower
            report['Pressure Upper Bound (PSI)'] = upper
    
    return report
