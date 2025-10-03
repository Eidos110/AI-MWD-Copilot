import shap
import matplotlib.pyplot as plt

def explain_model(model, X_sample, feature_names, title="SHAP Summary"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()