from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title_suffix: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusion {title_suffix}".strip())
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, title_suffix: str = "") -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"Curva ROC {title_suffix}".strip())
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()  # Muestra el gráfico en la celda
    return roc_auc


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str = "") -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    # Pasamos el nombre del modelo (si existe) para que aparezca en el título
    suffix = f"- {model_name}" if model_name else ""

    metrics["auc_roc"] = plot_roc_curve(y_true, y_prob, suffix)
    plot_confusion_matrix(y_true, y_pred, suffix)

    return metrics
