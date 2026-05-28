from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT_DIR / "artifact" / "final-model" / "model.pkl"
DEFAULT_METADATA_PATH = ROOT_DIR / "artifact" / "model_input_metadata.json"

NON_MODEL_COLUMNS = {
    "explicabilidad",
    "reglas_criticas_activadas",
    "reglas_criticas_activadas_json",
    "alertas_score_activadas",
    "ids_siniestros_similares_top5",
    "ids_siniestros_similares_top5_json",
}


def load_final_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Carga el modelo final exportado en artifact/final-model/model.pkl."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo final en: {model_path}")
    return joblib.load(model_path)


def load_model_metadata(metadata_path: str | Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """Carga metadata de entrenamiento, incluyendo columnas esperadas por el modelo."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _default_value_for_column(column: str, metadata: dict[str, Any]) -> Any:
    numeric_cols = set(metadata.get("numeric_cols", []))
    categorical_cols = set(metadata.get("categorical_cols", []))

    if column in numeric_cols:
        return 0
    if column in categorical_cols:
        return "Sin dato"
    if column.startswith("RF_") or column.startswith("score_") or column.startswith("freq_"):
        return 0
    return 0


def prepare_model_input(features_df: pd.DataFrame, metadata: dict[str, Any] | None = None) -> pd.DataFrame:
    """Ordena y completa las columnas de entrada que espera el pipeline del modelo."""
    if features_df.empty:
        return pd.DataFrame()

    metadata = metadata or {}
    expected_columns = metadata.get("model_input_columns")

    if not expected_columns:
        excluded = set(metadata.get("target_col", [])) | NON_MODEL_COLUMNS
        expected_columns = [col for col in features_df.columns if col not in excluded]

    model_input = features_df.copy()

    for column in expected_columns:
        if column not in model_input.columns:
            model_input[column] = _default_value_for_column(column, metadata)

    model_input = model_input[expected_columns].copy()

    for column in metadata.get("numeric_cols", []):
        if column in model_input.columns:
            model_input[column] = pd.to_numeric(model_input[column], errors="coerce").fillna(0)

    for column in metadata.get("categorical_cols", []):
        if column in model_input.columns:
            model_input[column] = model_input[column].fillna("Sin dato").astype(str)

    return model_input


def classify_claims(
    features_df: pd.DataFrame,
    model: Any | None = None,
    metadata: dict[str, Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Clasifica siniestros y agrega probabilidad_ml y prediccion_ml.

    Parameters
    ----------
    features_df:
        DataFrame ya procesado con build_features e integrate_rules_with_features.
    model:
        Modelo/pipeline cargado. Si no se entrega, se carga desde model_path.
    metadata:
        Metadata del entrenamiento. Si no se entrega, se carga desde metadata_path.
    threshold:
        Umbral para convertir probabilidad en prediccion binaria cuando el modelo expone predict_proba.
    """
    classified = features_df.copy()

    if classified.empty:
        classified["probabilidad_ml"] = pd.Series(dtype=float)
        classified["prediccion_ml"] = pd.Series(dtype=int)
        return classified

    metadata = metadata if metadata is not None else load_model_metadata(metadata_path)
    model = model if model is not None else load_final_model(model_path)
    X = prepare_model_input(classified, metadata)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            risk_probability = probabilities[:, 1]
        else:
            risk_probability = probabilities.ravel()
        classified["probabilidad_ml"] = pd.Series(risk_probability, index=classified.index).clip(0, 1)
        classified["prediccion_ml"] = (classified["probabilidad_ml"] >= threshold).astype(int)
    else:
        predictions = model.predict(X)
        classified["prediccion_ml"] = pd.Series(predictions, index=classified.index).astype(int)
        classified["probabilidad_ml"] = classified["prediccion_ml"].astype(float)

    return classified


# Alias en español para uso directo desde API/app.
def clasificar(
    features_df: pd.DataFrame,
    model: Any | None = None,
    metadata: dict[str, Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Alias de classify_claims."""
    return classify_claims(
        features_df=features_df,
        model=model,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path,
        threshold=threshold,
    )
