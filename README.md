# 🏢 AI MWD Copilot  
## Advanced Subsurface for Real-Time Drilling Intelligence System
### Internal R&D Prototype | Version 1.0  
**Prepared by**: Eidos/W_Isnal, Data Science & Geophysics  
**Date**: Oct 2025  
**Project ID**: DS-GEO-AI-2025-002  

---

## 🔍 Executive Summary

The **AI MWD Copilot** is a prototype machine learning system developed to enhance real-time decision-making during drilling operations by predicting key subsurface properties — **porosity**, **fluid type**, and **pore pressure** — using only available MWD/LWD telemetry.

Unlike conventional ML models that suffer from data leakage (e.g., using porosity logs to predict porosity), this system enforces **causal consistency** by isolating input features based on sensor physics and operational constraints.

This ensures predictions remain reliable even under degraded conditions (e.g., tool failure, missing sensors), making it suitable for deployment in field operations.

---

## 🎯 Business & Operational Objectives

| Goal | Benefit |
|------|--------|
| Reduce reliance on post-job analysis | Faster decisions, lower NPT |
| Predict formation properties when sensors fail | Increased robustness |
| Flag hydrocarbon zones early | Improve reservoir targeting |
| Estimate pore pressure trends in real time | Enhance wellbore stability |
| Deliver explainable outputs | Build trust with engineers |

This system supports strategic initiatives in **drilling digitalization**, **predictive analytics**, and **autonomous operations**.

---

## 🛠️ Technical Overview

### 1. Data Source
- High-frequency MWD/LWD telemetry dataset
- Includes: gamma ray, resistivity, density, neutron, drilling dynamics, gas chromatography, CoPilot-style diagnostics
- Sample rate: Depth-indexed (~0.1–0.3 m intervals)

### 2. Target Variables
| Property | Method |
|--------|--------|
| **Porosity** | Composite from density + neutron logs (Wyllie + RHG crossplot) |
| **Fluid Type** | Rule-based classification using resistivity, gas, and porosity thresholds |
| **Pore Pressure** | Rehm & McClendon method via corrected drilling exponent |

All targets validated against expected geological trends.

### 3. Model Design Principles
To ensure operational realism, the following principles were enforced:

| Principle | Implementation |
|---------|----------------|
| **No Data Leakage** | Porosity model excludes density/neutron logs |
| **Sensor Hierarchy** | Only causally valid inputs used per model |
| **Failure Mode Readiness** | Models tested under simulated sensor loss |
| **Interpretability** | SHAP used to explain predictions |
| **Modularity** | Clean separation of concerns (data → model → viz) |

### 4. Machine Learning Stack
- **Algorithms**: XGBoost (regression & classification)
- **Validation**: 80/20 train-test split, stratified where applicable
- **Metrics**:
  - Porosity: RMSE = 0.038
  - Fluid: F1-score (weighted) = 0.93
  - Pore Pressure: R² = 0.82, RMSE = 48.58 
- **Tools**: Python, scikit-learn, XGBoost, SHAP, Streamlit

---

## 📊 Performance Summary

| Model | Input Features | Key Drivers | RMSE / F1 | Notes |
|------|----------------|------------|--------|-------|
| **Porosity Predictor** | GR, Resistivity, ROP, WOB, Gas | Gamma Ray, Resistivity | RMSE = 0.038 | No direct porosity logs used |
| **Fluid Classifier** | GR, MSE, ROP, Stick-Slip | Gas trend, resistivity rise | F1 = 0.93 | Flags pay zones before GC confirmation |
| **Pore Pressure Estimator** | Mud Weight, ECD, ROP, WOB | ECD gradient, ROP drop | RMSE = 48.58 | Alternative to drilling exponent |

SHAP analysis confirms geologically consistent behavior across all models.

---

## 🖥️ Dashboard Interface

An interactive web dashboard was developed using **Streamlit** to visualize:
- Real-time strip log display
- AI-predicted vs. true values
- Fluid classification alerts
- SHAP-based explanation panel

Designed for use by:
- Drilling Engineers
- Petrophysicists
- On-site Decision Support Teams

![Dashboard](assets/screenshot.png)

---

## 🗂️ Deployment Architecture
```
[Raw MWD Data]
↓
[Data Preprocessor] → [Feature Isolator]
↓
[Prediction Engine] → [Porosity Model]
[Fluid Model]
[Pressure Model]
↓
[Visualization Layer] → Streamlit Dashboard

```
All components modular, container-ready, and scalable.


---

## 🧩 Project Structure
```
ai-mwd-copilot/
├── data/                 # Input datasets
├── models/               # Trained models (.pkl)
├── src/                  # Core modules
│ ├── config.py           # Central configuration
│ ├── data_loader.py      # Safe loading
│ ├── models.py           # Model manager
│ ├── plots.py            # Log visualization
│ └── shap_explainer.py   # Interpretability
├── app.py                # Dashboard entry point
├── requirements.txt      # Dependencies
└── README.md             # This document
```
---


## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Eidos110/ai-mwd-copilot.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Launch the dashboard:
   ```bash
   streamlit run app.py


---

## Future Roadmap
| INITIATIVE | STATUS |
|-----|-----|
| LSTM-based sequence modeling  | In design |
| REST API for real-time inference | Planned |
| Docker containerization | Next phase |
| Integration with existing D&A platforms | Feasibility study |
| Field trial simulation mode | Under development |


---

## 🙌 Acknowledgments
This prototype was developed independently using open-source tools and simulated industrial data patterns. Inspired by technologies from Schlumberger (OnTrak™, CoPilot®), Halliburton (Geo-Pilot®), and Baker Hughes.

Special thanks to the open-source community behind:

XGBoost, SHAP, Streamlit, scikit-learn


---

## 🏁 Disclaimers
 - This is a prototype and not intended for live well control.
 - Results are based on synthetic or anonymized data.

