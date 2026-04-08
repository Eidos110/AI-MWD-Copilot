"""
Data quality analysis module.

Provides utilities to assess dataset health:
- Missing value patterns
- Outlier detection
- Sensor health scores
- Data completeness metrics
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values per column.
    
    Args:
        df (pd.DataFrame): Input data.
    
    Returns:
        pd.DataFrame: Summary with columns:
            - 'Column': column name
            - 'Missing Count': number of NaN values
            - 'Missing %': percentage of missing data
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    return pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_count.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing %', ascending=False)


def detect_outliers(df: pd.DataFrame, numeric_cols: List[str] = None, iqr_multiplier: float = 1.5) -> Dict:
    """Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input data.
        numeric_cols (list, optional): Columns to analyze. Defaults to all numeric.
        iqr_multiplier (float): IQR multiplier for outlier bounds (default 1.5).
    
    Returns:
        dict: Outlier statistics per column with keys:
            - 'Lower Bound': Q1 - multiplier*IQR
            - 'Upper Bound': Q3 + multiplier*IQR
            - 'Outlier Count': number of outliers
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = outlier_mask.sum()
        
        outliers[col] = {
            'Lower Bound': lower,
            'Upper Bound': upper,
            'Outlier Count': outlier_count,
            'Outlier %': (outlier_count / len(df)) * 100
        }
    
    return outliers


def compute_data_completeness(df: pd.DataFrame, critical_cols: List[str] = None) -> Dict:
    """Compute overall data completeness score.
    
    Args:
        df (pd.DataFrame): Input data.
        critical_cols (list, optional): Columns critical for analysis.
    
    Returns:
        dict with keys:
            - 'Overall Completeness %': fraction of non-NaN values (0-100)
            - 'Critical Columns Completeness %': avg for critical cols
            - 'Rows with Complete Data': count of rows with no NaN
            - 'Rows with Any Missing': count of rows with >=1 NaN
    """
    overall = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
    
    complete_rows = (~df.isna().any(axis=1)).sum()
    incomplete_rows = df.isna().any(axis=1).sum()
    
    result = {
        'Overall Completeness %': round(overall, 2),
        'Rows with Complete Data': int(complete_rows),
        'Rows with Any Missing': int(incomplete_rows),
        'Total Rows': len(df)
    }
    
    if critical_cols:
        critical_present = [c for c in critical_cols if c in df.columns]
        if critical_present:
            critical_complete = (df[critical_present].notna().sum().sum() / 
                               (len(df) * len(critical_present))) * 100
            result['Critical Columns Completeness %'] = round(critical_complete, 2)
    
    return result


def compute_sensor_health(df: pd.DataFrame, feature_groups: Dict[str, List[str]] = None) -> pd.DataFrame:
    """Compute health score per sensor group (feature group).
    
    Health = (available features / total expected) * (1 - outlier_fraction)
    
    Args:
        df (pd.DataFrame): Input data.
        feature_groups (dict, optional): Mapping sensor name -> list of columns.
                                        If None, treats all numeric as one group.
    
    Returns:
        pd.DataFrame: Sensor health summary with columns:
            - 'Sensor': sensor/group name
            - 'Available Features': count of non-NaN columns
            - 'Outlier %': percentage of outlier rows in this group
            - 'Health Score': composite score (0-100)
    """
    if feature_groups is None:
        feature_groups = {'All Data': df.select_dtypes(include=[np.number]).columns.tolist()}
    
    results = []
    for sensor_name, cols in feature_groups.items():
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue
        
        # Feature availability
        available = (~df[valid_cols].isna()).mean().mean() * 100
        
        # Outlier detection for this group
        outliers = detect_outliers(df, valid_cols, iqr_multiplier=1.5)
        outlier_rows = set()
        for col, out_info in outliers.items():
            if out_info['Outlier Count'] > 0:
                col_outliers = (
                    (df[col] < out_info['Lower Bound']) | 
                    (df[col] > out_info['Upper Bound'])
                ).sum()
                outlier_rows.update(df[col][(df[col] < out_info['Lower Bound']) | 
                                           (df[col] > out_info['Upper Bound'])].index)
        
        outlier_pct = (len(outlier_rows) / len(df)) * 100 if len(df) > 0 else 0
        
        # Health score: availability * (1 - outlier fraction)
        health_score = available * (1 - outlier_pct / 100)
        
        results.append({
            'Sensor': sensor_name,
            'Available Features %': round(available, 2),
            'Outlier Rows %': round(outlier_pct, 2),
            'Health Score (0-100)': round(health_score, 2)
        })
    
    return pd.DataFrame(results).sort_values('Health Score (0-100)', ascending=False)


def generate_quality_report(df: pd.DataFrame, critical_cols: List[str] = None, 
                           feature_groups: Dict[str, List[str]] = None) -> Dict:
    """Generate comprehensive data quality report.
    
    Args:
        df (pd.DataFrame): Input data.
        critical_cols (list, optional): Critical columns for analysis.
        feature_groups (dict, optional): Sensor/feature grouping.
    
    Returns:
        dict with keys:
            - 'missing_values': DataFrame of missing value analysis
            - 'completeness': dict of completeness metrics
            - 'sensor_health': DataFrame of sensor health scores
            - 'summary': str with key findings
    """
    missing = analyze_missing_values(df)
    completeness = compute_data_completeness(df, critical_cols)
    health = compute_sensor_health(df, feature_groups)
    
    # Generate summary
    summary = f"""
**Data Quality Summary**

- Total Records: {len(df):,}
- Total Columns: {len(df.columns)}
- Overall Completeness: {completeness['Overall Completeness %']:.1f}%
- Rows with Complete Data: {completeness['Rows with Complete Data']}
- Rows with Missing Values: {completeness['Rows with Any Missing']}

**Columns with Highest Missing %:**
"""
    for idx, row in missing.head(3).iterrows():
        if row['Missing Count'] > 0:
            summary += f"\n- {row['Column']}: {row['Missing %']:.1f}% ({int(row['Missing Count'])} values)"
    
    summary += "\n\n**Sensor Health Scores:**\n"
    for idx, row in health.iterrows():
        status = "🟢 Good" if row['Health Score (0-100)'] >= 80 else \
                 "🟡 Fair" if row['Health Score (0-100)'] >= 60 else \
                 "🔴 Poor"
        summary += f"\n- {row['Sensor']}: {row['Health Score (0-100)']:.1f}/100 {status}"
    
    return {
        'missing_values': missing,
        'completeness': completeness,
        'sensor_health': health,
        'summary': summary
    }
