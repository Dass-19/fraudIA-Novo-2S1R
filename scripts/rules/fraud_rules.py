from __future__ import annotations

import json

import numpy as np
import pandas as pd

CRITICAL_RULES: dict[str, dict[str, str]] = {
    "RF_01_perdida_total_robo": {
        "nivel": "Rojo",
        "descripcion": "Cobertura de perdida total por robo.",
    },
    "RF_02_adulteracion_doc": {
        "nivel": "Rojo",
        "descripcion": "Evidencia de falsificacion, adulteracion o inconsistencia documental.",
    },
    "RF_03_lista_restrictiva": {
        "nivel": "Rojo",
        "descripcion": "Asegurado, beneficiario, APS o proveedor asociado a lista restrictiva u observada.",
    },
    "RF_04_dinamica_imposible": {
        "nivel": "Rojo",
        "descripcion": "Dinamica del accidente fisicamente imposible o relato ilogico.",
    },
    "RF_05_borde_vigencia_48h": {
        "nivel": "Amarillo",
        "descripcion": "Siniestro extremo al borde de vigencia menor o igual a 48 horas.",
    },
    "RF_06_demora_robo_4dias": {
        "nivel": "Amarillo",
        "descripcion": "Demora atipica de denuncia de robo superior a 4 dias.",
    },
    "RF_07_narrativa_clonada": {
        "nivel": "Amarillo",
        "descripcion": "Narrativa clonada o altamente similar a otro reclamo.",
    },
    "RF_08_score_reglas_alto": {
        "nivel": "Rojo",
        "descripcion": "Score total de reglas en nivel rojo.",
    },
    "RF_09_score_alto_y_ml_riesgo": {
        "nivel": "Rojo",
        "descripcion": "Score medio/alto combinado con prediccion del modelo ML en riesgo.",
    },
    "RF_10_documental_multiple": {
        "nivel": "Rojo",
        "descripcion": "Documentos incompletos e inconsistentes en el mismo siniestro.",
    },
    "RF_11_proveedor_recurrente_monto_atipico": {
        "nivel": "Rojo",
        "descripcion": "Proveedor recurrente u observado combinado con monto atipico.",
    },
    "RF_12_alta_frecuencia_y_borde_vigencia": {
        "nivel": "Amarillo",
        "descripcion": "Alta frecuencia de reclamos combinada con siniestro cercano a la vigencia.",
    },
}

ROJO_RULES = [key for key, meta in CRITICAL_RULES.items() if meta["nivel"] == "Rojo"]
AMARILLO_RULES = [key for key, meta in CRITICAL_RULES.items() if meta["nivel"] == "Amarillo"]


def _series(df: pd.DataFrame, column: str, default=0) -> pd.Series:
    value = df.get(column)
    if value is None:
        return pd.Series(default, index=df.index)
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=df.index)


def _bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    value = _series(df, column, default)
    if value.dtype == bool:
        return value.fillna(default)
    return value.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "si": True, "sí": True, "yes": True, "false": False, "0": False, "no": False}
    ).fillna(default)


def _num_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_series(df, column, default), errors="coerce").fillna(default)


def _min_days_to_policy_edge(df: pd.DataFrame) -> pd.Series:
    start_days = _num_series(df, "dias_desde_inicio_poliza", np.nan)
    end_days = _num_series(df, "dias_desde_fin_poliza", np.nan).abs()

    if start_days.isna().all() and {"fecha_ocurrencia", "fecha_inicio"}.issubset(df.columns):
        start_days = (pd.to_datetime(df["fecha_ocurrencia"], errors="coerce") - pd.to_datetime(df["fecha_inicio"], errors="coerce")).dt.total_seconds() / 86400
    if end_days.isna().all() and {"fecha_fin", "fecha_ocurrencia"}.issubset(df.columns):
        end_days = (pd.to_datetime(df["fecha_fin"], errors="coerce") - pd.to_datetime(df["fecha_ocurrencia"], errors="coerce")).dt.total_seconds().abs() / 86400

    return pd.concat([start_days, end_days], axis=1).min(axis=1)


def apply_critical_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Evalua reglas criticas y de advertencia basadas en los puntos 7 y 8 del reto."""
    rules_df = pd.DataFrame(index=df.index)

    cobertura = _series(df, "cobertura", "").astype(str).str.lower()
    descripcion = _series(df, "descripcion", "").astype(str).str.lower()
    min_days = _min_days_to_policy_edge(df)
    score_total = _num_series(df, "score_total_reglas", 0)
    pred_ml = _num_series(df, "prediccion_ml", 0)
    prob_ml = _num_series(df, "probabilidad_ml", 0)

    rules_df["RF_01_perdida_total_robo"] = cobertura.str.contains("perdida total|p[eé]rdida total|ptxrb", regex=True, na=False) & cobertura.str.contains("robo", na=False)
    doc_obs = _series(df, "docs_observaciones", "").astype(str).str.lower()
    explicit_doc_alteration = doc_obs.str.contains("fals|adulter|alteraci|manipulad|factura previa", regex=True, na=False)
    rules_df["RF_02_adulteracion_doc"] = explicit_doc_alteration
    rules_df["RF_03_lista_restrictiva"] = (_num_series(df, "score_proveedor", 0) >= 10) | (_num_series(df, "porcentaje_casos_observados", 0) >= 40)
    rules_df["RF_04_dinamica_imposible"] = descripcion.str.contains("imposible|ilogic|inconsisten|contradic|fisicamente imposible", regex=True, na=False)
    rules_df["RF_05_borde_vigencia_48h"] = min_days <= 2
    rules_df["RF_06_demora_robo_4dias"] = cobertura.str.contains("robo", na=False) & (_num_series(df, "dias_entre_ocurrencia_reporte", 0) > 4)
    rules_df["RF_07_narrativa_clonada"] = _num_series(df, "max_similitud_textual", 0) > 0.85
    rules_df["RF_08_score_reglas_alto"] = score_total >= 76
    rules_df["RF_09_score_alto_y_ml_riesgo"] = (score_total >= 41) & ((pred_ml == 1) | (prob_ml >= 0.70))
    rules_df["RF_10_documental_multiple"] = (
        (_num_series(df, "score_docs_incompletos", 0) > 0)
        & (_num_series(df, "score_docs_inconsistentes", 0) > 0)
        & ((pred_ml == 1) | (prob_ml >= 0.70) | (score_total >= 41))
    )
    rules_df["RF_11_proveedor_recurrente_monto_atipico"] = (
        (_num_series(df, "score_proveedor", 0) >= 5)
        & (_num_series(df, "score_monto_suma_asegurada", 0) >= 4)
        & ((pred_ml == 1) | (score_total >= 41))
    )
    rules_df["RF_12_alta_frecuencia_y_borde_vigencia"] = (
        (_num_series(df, "score_freq_asegurado", 0) >= 4)
        | (_num_series(df, "score_freq_vehiculo", 0) >= 3)
        | (_num_series(df, "score_freq_conductor", 0) >= 4)
    ) & (_num_series(df, "score_reclamo_vigencia", 0) >= 4)

    condicion_rojo = rules_df[ROJO_RULES].any(axis=1)
    condicion_amarillo = rules_df[AMARILLO_RULES].any(axis=1)
    rules_df["semaforo_reglas_criticas"] = np.where(condicion_rojo, "Rojo", np.where(condicion_amarillo, "Amarillo", "Verde"))
    rules_df["reglas_criticas_activadas"] = rules_df.apply(
        lambda row: [rule for rule in CRITICAL_RULES if bool(row.get(rule, False))],
        axis=1,
    )
    rules_df["reglas_criticas_activadas_json"] = rules_df["reglas_criticas_activadas"].apply(lambda items: json.dumps(items, ensure_ascii=False))
    return rules_df


def integrate_rules_with_features(features_df: pd.DataFrame) -> pd.DataFrame:
    rules_df = apply_critical_rules(features_df)
    cols_to_add = [c for c in rules_df.columns if c.startswith("RF_") or c in ["semaforo_reglas_criticas", "reglas_criticas_activadas", "reglas_criticas_activadas_json"]]
    base = features_df.drop(columns=[c for c in cols_to_add if c in features_df.columns], errors="ignore")
    return pd.concat([base, rules_df[cols_to_add]], axis=1)


def describe_activated_rules(rule_ids: list[str]) -> list[str]:
    return [CRITICAL_RULES[rule_id]["descripcion"] for rule_id in rule_ids if rule_id in CRITICAL_RULES]
