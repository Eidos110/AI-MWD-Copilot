"""Interpretation service for drilling decision support.

Provides functions to interpret ML predictions into actionable insights:
- Zone summary based on fluid classification
- Petrophysical interpretation (porosity, pressure)
- Drilling decision support recommendations
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Fix for deprecated np.int in newer numpy versions
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

import pandas as pd


POROSITY_QUALITY_THRESHOLDS = [
    (0.20, "Excellent"),
    (0.15, "Good"),
    (0.10, "Fair"),
    (0.05, "Poor"),
    (0.0, "Tight"),
]

FLUID_CLASS_LABELS = {
    "Potential Reservoir": "Potential Reservoir",
    "Pay Zone": "Pay Zone",
    "Background": "Background",
}

ZONE_RECOMMENDATIONS = {
    "Potential Reservoir": "Consider completion testing and production evaluation",
    "Pay Zone": "Evaluate with additional logs, may require further testing",
    "Background": "No action required - non-reservoir section",
}

PRESSURE_RISK_LEVELS = [
    (8000, "HIGH", "Abnormal high pore pressure - kicks likely"),
    (6000, "ELEVATED", "Abnormal pressure - may require mud weight adjustment"),
    (4000, "NORMAL", "Normal pressure gradient - continue with current parameters"),
    (0, "LOW", "Underbalanced - possible lost circulation"),
]

MUD_WEIGHT_FACTOR = 0.052


def _classify_porosity_quality(phi: float) -> str:
    """Classify porosity quality based on value."""
    for threshold, quality in POROSITY_QUALITY_THRESHOLDS:
        if phi >= threshold:
            return quality
    return "Tight"


def _get_porosity_quality_distribution(
    porosity: np.ndarray,
) -> Dict[str, int]:
    """Get distribution of porosity quality in dataset."""
    distribution = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0, "Tight": 0}
    for phi in porosity:
        if np.isnan(phi):
            continue
        quality = _classify_porosity_quality(float(phi))
        distribution[quality] = distribution.get(quality, 0) + 1
    return distribution


def _get_pressure_risk(pore_pressure: float) -> Tuple[str, str]:
    """Get pressure risk level and description."""
    if pore_pressure >= 8000:
        return "HIGH", "Abnormal high pore pressure - kicks likely"
    elif pore_pressure >= 6000:
        return "ELEVATED", "Abnormal pressure - may require mud weight adjustment"
    elif pore_pressure >= 4000:
        return "NORMAL", "Normal pressure gradient - continue with current parameters"
    else:
        return "LOW", "Underbalanced - possible lost circulation"


def calculate_mud_weight(
    depth: float, pore_pressure: float, safety_factor: float = 0.5
) -> float:
    """Calculate required mud weight in ppg."""
    if depth <= 0 or pore_pressure <= 0:
        return 0.0
    gradient = pore_pressure / depth
    required_mwd = gradient + safety_factor
    return round(required_mwd, 1)


def interpret_zones(
    df: pd.DataFrame,
    fluid_predictions: List[str],
    fluid_probabilities: Optional[np.ndarray] = None,
    confidence: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Interpret zones based on fluid predictions.

    Args:
        df: DataFrame with DEPTH column
        fluid_predictions: List of fluid class predictions
        fluid_probabilities: Optional probabilities from classifier
        confidence: Optional confidence scores

    Returns:
        Zone interpretation with recommendations
    """
    if len(df) == 0 or len(fluid_predictions) == 0:
        return {
            "zones": [],
            "summary": {
                "total_reservoir_ft": 0,
                "total_pay_zone_ft": 0,
                "total_background_ft": 0,
            },
        }

    depths = df["DEPTH"].values
    zones = []
    thickness_by_type = {"Potential Reservoir": 0.0, "Pay Zone": 0.0, "Background": 0.0}

    current_zone_type = fluid_predictions[0]
    zone_start = depths[0]

    for i in range(1, len(fluid_predictions)):
        if fluid_predictions[i] != current_zone_type:
            zone_end = depths[i]
            thickness = zone_end - zone_start

            avg_prob = 0.0
            if fluid_probabilities is not None:
                zone_probs = fluid_probabilities[:i]
                if len(zone_probs) > 0:
                    avg_prob = float(np.max(zone_probs, axis=1).mean())

            conf = confidence[i - 1] if confidence else avg_prob

            evidence = []
            if current_zone_type == "Potential Reservoir":
                if "GR" in df.columns:
                    gr_vals = df["GR"].values[:i]
                    if len(gr_vals) > 0 and np.nanmean(gr_vals) < 50:
                        evidence.append("Low GR (shale indicator)")
                if "Resistivity Phase - Corrected - 2MHz ohm.m" in df.columns:
                    res_col = "Resistivity Phase - Corrected - 2MHz ohm.m"
                    res_vals = df[res_col].values[:i]
                    if len(res_vals) > 0 and np.nanmean(res_vals) > 100:
                        evidence.append(
                            f"High resistivity ({np.nanmean(res_vals):.0f} ohm.m)"
                        )

            zones.append(
                {
                    "depth_start": float(zone_start),
                    "depth_end": float(zone_end),
                    "zone_type": current_zone_type,
                    "confidence": round(conf, 2) if conf else 0.0,
                    "thickness": round(thickness, 1),
                    "recommendation": ZONE_RECOMMENDATIONS.get(
                        current_zone_type, "No recommendation"
                    ),
                    "evidence": evidence,
                }
            )

            thickness_by_type[current_zone_type] = (
                thickness_by_type.get(current_zone_type, 0.0) + thickness
            )

            current_zone_type = fluid_predictions[i]
            zone_start = zone_end

    if len(fluid_predictions) > 0:
        zone_end = depths[-1]
        thickness = zone_end - zone_start
        thickness_by_type[current_zone_type] = (
            thickness_by_type.get(current_zone_type, 0.0) + thickness
        )

    return {
        "zones": zones,
        "summary": {
            "total_reservoir_ft": round(
                thickness_by_type.get("Potential Reservoir", 0.0), 1
            ),
            "total_pay_zone_ft": round(thickness_by_type.get("Pay Zone", 0.0), 1),
            "total_background_ft": round(thickness_by_type.get("Background", 0.0), 1),
        },
    }


def interpret_petrophysics(
    df: pd.DataFrame,
    porosity_predictions: Optional[List[float]] = None,
    pore_pressure_predictions: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Interpret petrophysical properties.

    Args:
        df: DataFrame with well data
        porosity_predictions: Optional porosity predictions
        pore_pressure_predictions: Optional pore pressure predictions

    Returns:
        Petrophysical interpretation
    """
    result = {
        "porosity_analysis": {
            "average": None,
            "quality": "Unknown",
            "distribution": {},
        },
        "pressure_analysis": {
            "normal": True,
            "anomaly_detected": False,
            "overbalance_risk": "Low",
        },
        "reservoir_quality": "Unknown",
        "hydrocarbon_potential": "Unknown",
    }

    if porosity_predictions:
        phi_arr = np.array(porosity_predictions)
        phi_avg = float(np.nanmean(phi_arr))
        phi_std = float(np.nanstd(phi_arr))

        result["porosity_analysis"]["average"] = round(phi_avg, 3)
        result["porosity_analysis"]["std_deviation"] = round(phi_std, 3)
        result["porosity_analysis"]["distribution"] = (
            _get_porosity_quality_distribution(phi_arr)
        )

        if phi_avg >= 0.20:
            result["porosity_analysis"]["quality"] = "Excellent"
            result["reservoir_quality"] = "Excellent"
        elif phi_avg >= 0.15:
            result["porosity_analysis"]["quality"] = "Good"
            result["reservoir_quality"] = "Good"
        elif phi_avg >= 0.10:
            result["porosity_analysis"]["quality"] = "Fair"
            result["reservoir_quality"] = "Fair"
        else:
            result["porosity_analysis"]["quality"] = "Poor"
            result["reservoir_quality"] = "Poor"

    if pore_pressure_predictions:
        pp_arr = np.array(pore_pressure_predictions)
        pp_avg = float(np.nanmean(pp_arr))
        pp_max = float(np.nanmax(pp_arr))

        normal_gradient = 0.433
        depth = df["DEPTH"].values
        if len(depth) > 0:
            depth_avg = float(np.nanmean(depth))
            expected_pp = normal_gradient * depth_avg * MUD_WEIGHT_FACTOR * 1000
            anomaly = pp_avg - expected_pp if expected_pp > 0 else 0

            risk_level, risk_desc = _get_pressure_risk(pp_avg)

            result["pressure_analysis"]["average_psi"] = round(pp_avg, 0)
            result["pressure_analysis"]["maximum_psi"] = round(pp_max, 0)
            result["pressure_analysis"]["risk_level"] = risk_level
            result["pressure_analysis"]["risk_description"] = risk_desc
            result["pressure_analysis"]["normal"] = risk_level == "NORMAL"
            result["pressure_analysis"]["anomaly_detected"] = risk_level != "NORMAL"
            result["pressure_analysis"]["overbalance_risk"] = risk_level

    reservoir_q = result.get("reservoir_quality", "Unknown")
    pressure_risk = result.get("pressure_analysis", {}).get("risk_level", "NORMAL")

    if reservoir_q in ["Excellent", "Good"] and pressure_risk in ["NORMAL", "LOW"]:
        result["hydrocarbon_potential"] = "High"
    elif reservoir_q == "Fair" and pressure_risk in ["NORMAL", "LOW"]:
        result["hydrocarbon_potential"] = "Moderate"
    elif pressure_risk == "HIGH":
        result["hydrocarbon_potential"] = "Low - pressure risk"
    else:
        result["hydrocarbon_potential"] = "Low"

    return result


def interpret_drilling(
    df: pd.DataFrame,
    pore_pressure_predictions: Optional[List[float]] = None,
    porosity_predictions: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Interpret for drilling decision support.

    Args:
        df: DataFrame with well data
        pore_pressure_predictions: Optional pore pressure predictions
        porosity_predictions: Optional porosity predictions

    Returns:
        Drilling decision support recommendations
    """
    result = {
        "drilling_conditions": [],
        "overall_assessment": "Normal drilling conditions",
        "critical_depths": [],
    }

    if pore_pressure_predictions is None or len(pore_pressure_predictions) == 0:
        return result

    depths = df["DEPTH"].values
    pp_arr = np.array(pore_pressure_predictions)

    drilling_conditions = []
    warnings = []
    critical_depths = []

    for i in range(len(pp_arr)):
        depth = float(depths[i])
        pp = float(pp_arr[i])

        risk_level, risk_desc = _get_pressure_risk(pp)
        mud_weight = calculate_mud_weight(depth, pp)

        depth_warnings = []
        if risk_level == "HIGH":
            warnings.append(f"High pore pressure at {depth:.0f} ft: {risk_desc}")
            critical_depths.append(
                {"depth": depth, "issue": "HIGH_PRESSURE", "pressure": pp}
            )
            depth_warnings.append("Consider increasing mud weight")
        elif risk_level == "ELEVATED":
            warnings.append(f"Elevated pore pressure at {depth:.0f} ft")
            depth_warnings.append("Monitor for kicks")

        if porosity_predictions and i < len(porosity_predictions):
            phi = porosity_predictions[i]
            if phi is not None and phi < 0.05:
                depth_warnings.append("Low porosity - tight zone")

        drilling_conditions.append(
            {
                "depth": depth,
                "pore_pressure_psi": round(pp, 0),
                "mud_weight_required_ppg": mud_weight,
                "kick_risk": risk_level,
                "recommendation": _get_drilling_recommendation(risk_level),
                "warnings": depth_warnings,
            }
        )

    result["drilling_conditions"] = drilling_conditions
    result["overall_assessment"] = _get_overall_assessment(warnings)
    result["critical_depths"] = critical_depths
    result["warnings"] = warnings

    return result


def _get_drilling_recommendation(risk_level: str) -> str:
    """Get drilling recommendation based on risk level."""
    recommendations = {
        "HIGH": "Stop drilling - increase mud weight before proceeding",
        "ELEVATED": "Proceed with caution - have kick detection equipment ready",
        "NORMAL": "Continue with current drilling parameters",
        "LOW": "Lost circulation risk - consider reducing mud weight",
    }
    return recommendations.get(risk_level, "Continue monitoring")


def _get_overall_assessment(warnings: List[str]) -> str:
    """Get overall drilling assessment."""
    if not warnings:
        return "Normal drilling conditions"

    high_count = sum(1 for w in warnings if "HIGH" in w or "High" in w)
    elevated_count = sum(1 for w in warnings if "ELEVATED" in w or "Elevated" in w)

    if high_count > 0:
        return f"Abnormal conditions - {high_count} critical depth(s) require attention"
    elif elevated_count > 0:
        return f"Monitoring required - {elevated_count} zone(s) with elevated pressure"
    else:
        return "Normal drilling conditions"


def generate_interpretation_report(
    df: pd.DataFrame,
    fluid_predictions: Optional[List[str]] = None,
    fluid_probabilities: Optional[np.ndarray] = None,
    porosity_predictions: Optional[List[float]] = None,
    pore_pressure_predictions: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Generate complete interpretation report.

    Args:
        df: DataFrame with well data
        fluid_predictions: Fluid class predictions
        fluid_probabilities: Fluid class probabilities
        porosity_predictions: Porosity predictions
        pore_pressure_predictions: Pore pressure predictions

    Returns:
        Complete interpretation report
    """
    report = {
        "zones": None,
        "petrophysics": None,
        "drilling": None,
    }

    if fluid_predictions:
        report["zones"] = interpret_zones(df, fluid_predictions, fluid_probabilities)

    report["petrophysics"] = interpret_petrophysics(
        df, porosity_predictions, pore_pressure_predictions
    )

    if pore_pressure_predictions:
        report["drilling"] = interpret_drilling(
            df, pore_pressure_predictions, porosity_predictions
        )

    return report
