from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pathlib import Path


def load_model(model_path: Path):
    return joblib.load(model_path)


def _get_explainer(model, X: pd.DataFrame):
    try:
        return shap.Explainer(model, X)
    except Exception:
        return shap.Explainer(model.predict_proba, X)


def explain_model(
    model,
    X: pd.DataFrame,
    max_samples: int = 200,
):

    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    preprocessor = model.named_steps['preprocess']
    classifier = model.steps[-1][1]

    X_sample_transformed = preprocessor.transform(X_sample)

    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
        if hasattr(X_sample_transformed, 'toarray'):
            X_sample_transformed = X_sample_transformed.toarray()

        X_sample_transformed = pd.DataFrame(
            X_sample_transformed,
            columns=feature_names
        )

    explainer = _get_explainer(classifier, X_sample_transformed)
    shap_values = explainer(X_sample_transformed)

    plt.figure()
    if len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    elif isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    shap.summary_plot(shap_values_to_plot, X_sample_transformed)

    plt.show()
