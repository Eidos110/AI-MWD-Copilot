"""Check shap dependency requirements"""
import sys, os
sys.path.insert(0, "E:\\Code\\Well-Logging-AI-AWD-Copilot-Deepseek")
try:
    import shap
    print("shap imported OK, version:", shap.__version__)
except Exception as e:
    print(f"shap import error: {type(e).__name__}: {e}")

try:
    import torch
    print("torch imported OK")
except ImportError:
    print("torch: NOT AVAILABLE (expected)")
except Exception as e:
    print(f"torch: {type(e).__name__}: {e}")
