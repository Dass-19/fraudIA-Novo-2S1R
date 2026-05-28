# AGENTS.md

Guia rapida del proyecto y el flujo de trabajo actual.

## Vision general
- Objetivo: clasificacion binaria de fraude en siniestros (`etiqueta_fraude_simulada`), con explicabilidad y scoring por reglas.
- Flujo principal: notebooks para preprocesamiento, modelado y evaluacion; scripts para funciones reutilizables.
- Exportaciones: modelos en `artifact/` y modelo final en `artifact/final-model/`.

## Estructura del proyecto
- `data/synthetic/`: datos sinteticos originales (CSV).
- `data/raw/`: tablas preprocesadas desde el notebook 01.
- `data/processed/`: features y salidas intermedias (predicciones, narrativas).
- `notebooks/`:
  - `01_exploracion_datos.ipynb`: preprocesamiento + EDA end-to-end.
  - `02_modelo_fraude.ipynb`: entrenamiento con POO y exportacion de modelos.
  - `03_evaluacion_modelo.ipynb`: metricas, SHAP y seleccion del mejor modelo.
- `scripts/features/build_features.py`: feature engineering por reglas y joins.
- `scripts/evaluation/metrics.py`: metricas de clasificacion + graficos.
- `scripts/text/similarity.py`: TF-IDF + cosine similarity para narrativas.
- `scripts/explainability/explain_score.py`: funciones SHAP reutilizables.
- `scripts/models/fraud_model.py`: carga del modelo final y prediccion.
- `artifact/`: modelos entrenados (estructura fija).
- `reports/`: salidas de metricas y explicabilidad.

## Flujo recomendado (paso a paso)
1) Ejecutar `notebooks/01_exploracion_datos.ipynb`
   - Genera `data/raw/*.csv`.
2) Ejecutar `scripts/features/build_features.py`
   - Genera `data/processed/features_siniestros.csv`.
3) Ejecutar `notebooks/02_modelo_fraude.ipynb`
   - Entrena Logistic Regression, Decision Tree, Random Forest.
   - Exporta modelos fijos en `artifact/`:
     - `artifact/logistic_regression.pkl`
     - `artifact/decision_tree.pkl`
     - `artifact/random_forest.pkl`
   - Genera `data/processed/predictions.csv`.
4) Ejecutar `scripts/text/similarity.py`
   - Genera `data/processed/narrativas_similares.csv`.
5) Ejecutar `notebooks/03_evaluacion_modelo.ipynb`
   - Usa `scripts/evaluation/metrics.py` y `scripts/explainability/explain_score.py`.
   - Exporta modelo final: `artifact/final-model/model.pkl`.

## Notas de modelado
- El notebook 02 usa pipelines con imputacion:
  - Numericas: mediana.
  - Categoricas: moda + one-hot.
- La seleccion del mejor modelo se hace por `f1_score` en el notebook 03.
- La explicabilidad SHAP se genera desde `scripts/explainability/explain_score.py`.

## Entradas y salidas clave
- Entrada principal de features: `data/processed/features_siniestros.csv`.
- Predicciones para evaluacion: `data/processed/predictions.csv`.
- Salidas de metricas: `reports/metrics/` (matriz de confusion y ROC).
- Salidas SHAP: `reports/explainability/`.

## Dependencias
- Requiere `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `joblib`.
- Se recomienda ejecutar dentro del `venv/` incluido en el repo.
