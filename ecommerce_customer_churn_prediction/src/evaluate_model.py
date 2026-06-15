import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve)


def display_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    return report


def plot_confusion_matrix(model, X_test, y_test, title: str = "Confusion Matrix"):
    y_pred = model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues",
        display_labels=["Stay", "Churn"],
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    return disp.figure_


def plot_roc_curve(model, X_test, y_test, title: str = "ROC Curve"):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_feature_importance(model, feature_names, top_n: int = 12):
    importance = None
    if hasattr(model.named_steps["classifier"], "feature_importances_"):
        importance = model.named_steps["classifier"].feature_importances_
    elif hasattr(model.named_steps["classifier"], "coef_"):
        importance = np.abs(model.named_steps["classifier"].coef_[0])

    if importance is None:
        raise ValueError("Model does not expose feature importance.")

    if hasattr(model.named_steps["preprocessor"], "transformers_"):
        transformed_names = []
        for name, trans, cols in model.named_steps["preprocessor"].transformers_:
            if name == "numeric":
                transformed_names.extend(cols)
            elif name == "categorical":
                encoder = trans.named_steps["encoder"]
                categories = encoder.categories_
                for col, cats in zip(cols, categories):
                    transformed_names.extend([f"{col}_{cat}" for cat in cats])
        feature_names = transformed_names

    importance_series = pd.Series(importance, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    importance_series.plot(kind="barh", color="#2c7fb8")
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    return plt.gcf()


def explain_with_shap(model, X_sample):
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP is not installed. Please add shap to requirements.txt to use explainability.")

    classifier = model.named_steps["classifier"]
    explainer = shap.Explainer(classifier, model.named_steps["preprocessor"].transform(X_sample))
    shap_values = explainer(model.named_steps["preprocessor"].transform(X_sample))
    shap.summary_plot(shap_values, feature_names=X_sample.columns, show=False)
    return shap_values
