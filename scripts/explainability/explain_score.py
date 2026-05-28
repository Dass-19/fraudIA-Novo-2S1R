from __future__ import annotations

import ast
import json
from typing import Any

import pandas as pd
from pandas.api.types import is_scalar

from scripts.features.build_features import SCORE_COLUMNS, SCORE_RULE_LABELS
from scripts.rules.fraud_rules import describe_activated_rules


def _coerce_float(value: Any, default: float = 0.0) -> float:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return default
    return float(numeric_value)


def _coerce_int(value: Any, default: int = 0) -> int:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return default
    return int(numeric_value)


def semaforo_from_score(value: float) -> str:
    """Convierte un score 0-100 en semaforo de riesgo."""
    numeric_value = _coerce_float(value)
    if numeric_value <= 40:
        return "Verde"
    if numeric_value <= 75:
        return "Amarillo"
    return "Rojo"


def upgrade_semaforo(base: str, reglas: str, pred_ml: int, score_final: float) -> str:
    """Ajusta el semaforo final combinando score, reglas criticas y prediccion ML."""
    if reglas == "Rojo":
        return "Rojo"
    if base == "Rojo":
        return "Rojo"
    if reglas == "Amarillo" or (int(pred_ml) == 1 and float(score_final) >= 41):
        return "Amarillo"
    return base


def add_score_final(
    df: pd.DataFrame,
    rule_weight: float = 0.70,
    ml_weight: float = 0.30,
    rule_score_col: str = "score_total_reglas",
    ml_probability_col: str = "probabilidad_ml",
    output_col: str = "score_final",
) -> pd.DataFrame:
    """Agrega score_final = 70% score de reglas + 30% probabilidad ML normalizada a 0-100."""
    scored = df.copy()
    score_reglas = pd.to_numeric(scored.get(rule_score_col, 0), errors="coerce").fillna(0).clip(0, 100)
    score_ml = pd.to_numeric(scored.get(ml_probability_col, 0), errors="coerce").fillna(0).clip(0, 1) * 100
    scored[output_col] = (rule_weight * score_reglas + ml_weight * score_ml).round(2)
    return scored


def add_final_semaforo(
    df: pd.DataFrame,
    score_final_col: str = "score_final",
    prediction_col: str = "prediccion_ml",
    critical_rules_col: str = "semaforo_reglas_criticas",
) -> pd.DataFrame:
    """Agrega semaforo_score_final y semaforo_final para consumo de notebooks o app."""
    scored = df.copy()
    scored["semaforo_score_final"] = scored[score_final_col].apply(semaforo_from_score)
    scored["semaforo_final"] = scored.apply(
        lambda row: upgrade_semaforo(
            row.get("semaforo_score_final", "Verde"),
            row.get(critical_rules_col, "Verde"),
            _coerce_int(row.get(prediction_col, 0)),
            _coerce_float(row.get(score_final_col, 0)),
        ),
        axis=1,
    )
    return scored


def safe_list(value: Any) -> list[Any]:
    """Normaliza listas reales o serializadas como string a list; valores vacios devuelven []."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if is_scalar(value) and pd.isna(value):
        return []
    if isinstance(value, str):
        clean_value = value.strip()
        if clean_value in {"", "nan", "None", "null"}:
            return []
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(clean_value)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                continue
    return []


def active_score_alerts(row: pd.Series) -> list[str]:
    """Devuelve etiquetas de las alertas del punto 7 cuyo score sea mayor a cero."""
    alerts: list[str] = []
    for col in SCORE_COLUMNS:
        value = pd.to_numeric(row.get(col, 0), errors="coerce")
        if pd.notna(value) and value > 0:
            alerts.append(SCORE_RULE_LABELS[col])
    return alerts


def build_explanation(row: pd.Series) -> str:
    """Genera explicabilidad basada en score, reglas criticas, ML y similitud textual."""
    score_alerts = active_score_alerts(row)[:4]
    rule_ids = safe_list(row.get("reglas_criticas_activadas", []))
    rule_desc = describe_activated_rules(rule_ids)[:3]

    score_final = _coerce_float(row.get("score_final", 0))
    score_total_reglas = _coerce_float(row.get("score_total_reglas", 0))
    probabilidad_ml = _coerce_float(row.get("probabilidad_ml", 0))
    max_similitud = _coerce_float(row.get("max_similitud_textual", 0))

    parts = [
        f"Caso clasificado como {row.get('semaforo_final', 'Sin clasificar')} con score final {score_final:.2f}.",
        f"El score de reglas aporta {score_total_reglas:.0f} puntos y el modelo ML estima probabilidad de riesgo {probabilidad_ml:.2%}.",
    ]
    if score_alerts:
        parts.append("Alertas principales del punto 7: " + "; ".join(score_alerts) + ".")
    if rule_desc:
        parts.append("Reglas criticas/advertencias activadas del punto 8: " + "; ".join(rule_desc) + ".")
    if max_similitud > 0:
        parts.append(f"La narrativa tiene similitud maxima de {max_similitud:.2f} con otros siniestros.")

    parts.append("La salida es una alerta para revision humana; no constituye acusacion ni decision automatica de fraude.")
    return " ".join(parts)


def add_explainability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas serializadas y explicabilidad final reutilizables en app o notebook."""
    scored = df.copy()

    scored["alertas_score_activadas"] = scored.apply(
        lambda row: json.dumps(active_score_alerts(row), ensure_ascii=False),
        axis=1,
    )

    if "reglas_criticas_activadas" in scored.columns:
        scored["reglas_criticas_activadas"] = scored["reglas_criticas_activadas"].apply(
            lambda value: json.dumps(safe_list(value), ensure_ascii=False)
        )
    else:
        scored["reglas_criticas_activadas"] = json.dumps([], ensure_ascii=False)

    if "ids_siniestros_similares_top5_json" in scored.columns:
        scored["ids_siniestros_similares_top5"] = scored["ids_siniestros_similares_top5_json"]
    elif "ids_siniestros_similares_top5" in scored.columns:
        scored["ids_siniestros_similares_top5"] = scored["ids_siniestros_similares_top5"].apply(
            lambda value: json.dumps(safe_list(value), ensure_ascii=False)
        )
    else:
        scored["ids_siniestros_similares_top5"] = json.dumps([], ensure_ascii=False)

    scored["explicabilidad"] = scored.apply(build_explanation, axis=1)
    return scored


def add_final_score_and_explanation(
    df: pd.DataFrame,
    rule_weight: float = 0.70,
    ml_weight: float = 0.30,
) -> pd.DataFrame:
    """Pipeline reutilizable: score final, semaforo final y explicabilidad."""
    scored = add_score_final(df, rule_weight=rule_weight, ml_weight=ml_weight)
    scored = add_final_semaforo(scored)
    scored = add_explainability_columns(scored)
    return scored
