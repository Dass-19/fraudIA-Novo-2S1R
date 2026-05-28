from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Estructura esperada para build_features(tables):
# tables = {
#     "siniestros": pd.DataFrame,
#     "polizas": pd.DataFrame,
#     "asegurados": pd.DataFrame,
#     "documentos": pd.DataFrame,
#     "beneficiarios_proveedores": pd.DataFrame,
#     "vehiculos": pd.DataFrame,
# }
# La carga y el guardado de CSV se hacen desde los notebooks para evitar rutas
# hardcodeadas dentro de la capa reutilizable.

SPANISH_STOP_WORDS = [
    "a", "aca", "ahi", "al", "algo", "algunas", "algunos", "ante", "antes",
    "como", "con", "contra", "cual", "de", "del", "desde", "e", "el", "ella",
    "ellas", "ellos", "en", "entre", "era", "eran", "esa", "esas", "ese", "eso",
    "esta", "estaba", "estaban", "este", "esto", "la", "las", "lo", "los", "mas",
    "me", "mi", "mis", "no", "nos", "o", "para", "pero", "por", "que", "se",
    "sin", "sobre", "su", "sus", "tambien", "te", "tu", "un", "una", "y",
]

SCORE_RULE_LABELS: dict[str, str] = {
    "score_reclamo_vigencia": "Siniestro cercano al inicio o fin de vigencia de la poliza",
    "score_demora_robo": "Demora relevante entre ocurrencia y denuncia en cobertura de robo",
    "score_freq_asegurado": "Alta frecuencia de reclamos del asegurado en los ultimos 18 meses",
    "score_freq_vehiculo": "Alta frecuencia de reclamos del vehiculo en los ultimos 18 meses",
    "score_freq_conductor": "Alta frecuencia del conductor/asegurado en siniestros recientes",
    "score_rc_only": "Frecuencia atipica de reclamos donde solo se afecta Responsabilidad Civil",
    "score_proveedor": "Beneficiario o proveedor recurrente u observado",
    "score_docs_incompletos": "Documentacion requerida incompleta",
    "score_docs_inconsistentes": "Documentos ilegibles, inconsistentes o con fechas no concordantes",
    "score_dinamica_sospechosa": "Narrativa o dinamica del accidente requiere revision minuciosa",
    "score_sin_tercero": "Evento sin tercero identificado o con tercero que huye/no se identifica",
    "score_reporte_tardio": "Reporte tardio del siniestro frente a la fecha de ocurrencia",
    "score_monto_suma_asegurada": "Monto reclamado cercano a la suma asegurada o superior al promedio esperado",
    "score_narrativas_similares": "Narrativa similar a otros siniestros reportados",
}

SCORE_COLUMNS = list(SCORE_RULE_LABELS.keys())


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _table(tables: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    df = tables.get(name, _empty_df())
    return df.copy()


def _series(df: pd.DataFrame, column: str, default_value: Any = np.nan) -> pd.Series:
    value = df.get(column)
    if value is None:
        return pd.Series(default_value, index=df.index)
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=df.index)


def _to_bool_series(value: pd.Series) -> pd.Series:
    if value.dtype == bool:
        return value.fillna(False)
    return value.astype(str).str.strip().str.lower().map(
        {
            "true": True,
            "1": True,
            "si": True,
            "sí": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
            "nan": False,
            "none": False,
        }
    ).fillna(False)


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_series(df, col, default), errors="coerce").fillna(default)


def score_reclamo_vigencia(df: pd.DataFrame) -> pd.Series:
    start_days = _safe_numeric(df, "dias_desde_inicio_poliza", np.nan)
    end_days = _safe_numeric(df, "dias_desde_fin_poliza", np.nan).abs()
    min_days = pd.concat([start_days, end_days], axis=1).min(axis=1)
    score = np.where(min_days <= 10, 8, np.where(min_days <= 30, 4, 0))
    return pd.Series(score, index=df.index).astype(int)


def score_demora_denuncia_robo(df: pd.DataFrame) -> pd.Series:
    cobertura = _series(df, "cobertura", "").astype(str)
    is_robo = cobertura.str.contains("robo", case=False, na=False)
    days = _safe_numeric(df, "dias_entre_ocurrencia_reporte", 0)
    score = np.where(days > 2, 8, np.where((days >= 1) & (days <= 2), 4, 0))
    return pd.Series(np.where(is_robo, score, 0), index=df.index).astype(int)


def score_reporte_tardio(df: pd.DataFrame) -> pd.Series:
    days = _safe_numeric(df, "dias_entre_ocurrencia_reporte", 0)
    score = np.where(days > 7, 5, np.where((days >= 4) & (days <= 7), 3, 0))
    return pd.Series(score, index=df.index).astype(int)


def score_frecuencia(counts: pd.Series, high_pts: int, mid_pts: int, high_thr: int = 3, mid_thr: int = 2) -> pd.Series:
    numeric_counts = pd.to_numeric(counts, errors="coerce").fillna(0)
    score = np.where(numeric_counts >= high_thr, high_pts, 0)
    score = np.where(numeric_counts == mid_thr, mid_pts, score)
    return pd.Series(score, index=counts.index).astype(int)


def score_rc_only(prior_counts: pd.Series) -> pd.Series:
    numeric_counts = pd.to_numeric(prior_counts, errors="coerce").fillna(0)
    score = np.where(numeric_counts > 2, 6, np.where(numeric_counts == 1, 3, 0))
    return pd.Series(score, index=prior_counts.index).astype(int)


def score_proveedor(df: pd.DataFrame) -> pd.Series:
    pct_obs = _safe_numeric(df, "porcentaje_casos_observados", 0)
    reclamos = _safe_numeric(df, "reclamos_asociados", 0)
    en_lista = pct_obs >= 40
    score = np.where(en_lista, 10, np.where(reclamos > 2, 5, 0))
    return pd.Series(score, index=df.index).astype(int)


def score_docs_incompletos(df: pd.DataFrame) -> pd.Series:
    docs_completos = _to_bool_series(_series(df, "documentos_completos", False))
    faltantes = _to_bool_series(_series(df, "docs_faltantes", False))
    score = np.where((~docs_completos) | faltantes, 4, 0)
    return pd.Series(score, index=df.index).astype(int)


def score_docs_inconsistentes(df: pd.DataFrame) -> pd.Series:
    inconsistentes = _to_bool_series(_series(df, "docs_inconsistentes", False))
    score = np.where(inconsistentes, 10, 0)
    return pd.Series(score, index=df.index).astype(int)


def score_dinamica_sospechosa(df: pd.DataFrame) -> pd.Series:
    desc = _series(df, "descripcion", "").astype(str)
    pattern = "atipic|observado|inconsisten|imposible|contradic|frontal|posterior|volcadura|multiple|madrugada"
    flag = desc.str.contains(pattern, case=False, regex=True, na=False)
    score = np.where(flag, 6, 0)
    return pd.Series(score, index=df.index).astype(int)


def score_evento_sin_tercero(df: pd.DataFrame) -> pd.Series:
    desc = _series(df, "descripcion", "").astype(str).str.lower()
    flag = desc.str.contains("fuga|huy[oó]|sin tercero|no identificado|tercero no", regex=True, na=False)
    return pd.Series(np.where(flag, 5, 0), index=df.index).astype(int)


def score_monto_suma_asegurada(df: pd.DataFrame) -> pd.Series:
    monto = _safe_numeric(df, "monto_reclamado", 0)
    suma = _safe_numeric(df, "suma_asegurada", np.nan).replace(0, np.nan)
    estimado = _safe_numeric(df, "monto_estimado", np.nan)
    mean_est = estimado.mean(skipna=True)
    ratio = monto / suma
    score = np.where(ratio > 0.95, 5, np.where(monto > 1.5 * mean_est, 4, 0))
    return pd.Series(score, index=df.index).fillna(0).astype(int)


def score_narrativas_similares(max_similarity: pd.Series) -> pd.Series:
    sims = pd.to_numeric(max_similarity, errors="coerce").fillna(0)
    score = np.where(sims > 0.85, 8, np.where((sims >= 0.70) & (sims <= 0.85), 4, 0))
    return pd.Series(score, index=max_similarity.index).astype(int)


def count_prev_events(df: pd.DataFrame, key_col: str, date_col: str, window_days: int = 548) -> pd.Series:
    if key_col not in df.columns or date_col not in df.columns or df.empty:
        return pd.Series(0, index=df.index, dtype=int)

    temp = df[[key_col, date_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    counts = pd.Series(0, index=temp.index, dtype=int)

    for _, group in temp.dropna(subset=[key_col]).groupby(key_col, sort=False):
        group = group.sort_values(date_col)
        valid_dates = group[date_col]
        for idx, current_date in valid_dates.items():
            if pd.isna(current_date):
                counts.loc[idx] = 0
                continue
            start = current_date - pd.Timedelta(days=window_days)
            counts.loc[idx] = int(((valid_dates < current_date) & (valid_dates >= start)).sum())

    return counts.reindex(df.index, fill_value=0).astype(int)


def build_narrative_similarity_features(
    df: pd.DataFrame,
    text_col: str = "descripcion",
    id_col: str = "id_siniestro",
    max_per_record: int = 5,
) -> pd.DataFrame:
    """Genera top 5 de siniestros mas similares por narrativa y maxima similitud."""
    if df.empty or text_col not in df.columns or id_col not in df.columns:
        return pd.DataFrame(
            {
                id_col: _series(df, id_col, pd.NA),
                "ids_siniestros_similares_top5": [[] for _ in range(len(df))],
                "max_similitud_textual": 0.0,
            },
            index=df.index,
        )

    text = df[text_col].fillna("").astype(str).str.strip()
    ids = df[id_col].to_numpy()

    if len(df) <= 1 or text.str.len().sum() == 0:
        return pd.DataFrame(
            {
                id_col: ids,
                "ids_siniestros_similares_top5": [[] for _ in range(len(df))],
                "max_similitud_textual": 0.0,
            },
            index=df.index,
        )

    vectorizer = TfidfVectorizer(stop_words=SPANISH_STOP_WORDS, ngram_range=(1, 2), min_df=1)
    try:
        tfidf = vectorizer.fit_transform(text)
        sim_matrix = cosine_similarity(tfidf)
    except ValueError:
        sim_matrix = np.zeros((len(df), len(df)), dtype=float)

    np.fill_diagonal(sim_matrix, 0.0)
    top_ids: list[list[Any]] = []
    max_sims: list[float] = []

    for i in range(sim_matrix.shape[0]):
        sims = sim_matrix[i]
        idx_sorted = np.argsort(sims)[::-1][:max_per_record]
        row_ids = [ids[j].item() if hasattr(ids[j], "item") else ids[j] for j in idx_sorted if sims[j] > 0]
        top_ids.append(row_ids[:max_per_record])
        max_sims.append(float(sims[idx_sorted[0]]) if len(idx_sorted) else 0.0)

    return pd.DataFrame(
        {
            id_col: ids,
            "ids_siniestros_similares_top5": top_ids,
            "max_similitud_textual": max_sims,
        },
        index=df.index,
    )


def _aggregate_documents(documentos: pd.DataFrame, siniestros: pd.DataFrame) -> pd.DataFrame:
    if documentos.empty or "id_siniestro" not in documentos.columns:
        return pd.DataFrame(columns=["id_siniestro", "docs_inconsistentes", "docs_faltantes", "docs_observaciones"])

    docs = documentos.copy()
    docs = _parse_dates(docs, ["fecha_emision"])
    docs["entregado_bool"] = _to_bool_series(_series(docs, "entregado", True))
    docs["legible_bool"] = _to_bool_series(_series(docs, "legible", True))
    docs["inconsistencia_bool"] = _to_bool_series(_series(docs, "inconsistencia_detectada", False))

    if "fecha_ocurrencia" in siniestros.columns:
        docs = docs.merge(siniestros[["id_siniestro", "fecha_ocurrencia"]], on="id_siniestro", how="left")
        docs["factura_previa"] = (
            _series(docs, "tipo_documento", "").astype(str).str.upper().eq("FACTURA")
            & (pd.to_datetime(_series(docs, "fecha_emision"), errors="coerce") < pd.to_datetime(_series(docs, "fecha_ocurrencia"), errors="coerce"))
        )
    else:
        docs["factura_previa"] = False

    docs["doc_inconsistente"] = docs["inconsistencia_bool"] | (~docs["legible_bool"]) | docs["factura_previa"].fillna(False)

    docs_agg = (
        docs.groupby("id_siniestro")
        .agg(
            docs_inconsistentes=("doc_inconsistente", "max"),
            docs_faltantes=("entregado_bool", lambda s: bool((~s.fillna(True)).any())),
            docs_observaciones=("observacion", lambda s: " | ".join([str(v) for v in s.dropna().unique()[:3]])),
        )
        .reset_index()
    )
    return docs_agg


def _prepare_tables(tables: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    siniestros = _table(tables, "siniestros")
    polizas = _table(tables, "polizas")
    asegurados = _table(tables, "asegurados")
    documentos = _table(tables, "documentos")
    proveedores = _table(tables, "beneficiarios_proveedores")
    vehiculos = _table(tables, "vehiculos")

    siniestros = _parse_dates(siniestros, ["fecha_ocurrencia", "fecha_reporte", "created_at", "updated_at"])
    polizas = _parse_dates(polizas, ["fecha_inicio", "fecha_fin", "created_at", "updated_at"])
    asegurados = _parse_dates(asegurados, ["created_at", "updated_at"])
    proveedores = _parse_dates(proveedores, ["created_at", "updated_at"])
    vehiculos = _parse_dates(vehiculos, ["created_at", "updated_at"])
    documentos = _parse_dates(documentos, ["fecha_emision", "created_at", "updated_at"])
    return siniestros, polizas, asegurados, documentos, proveedores, vehiculos


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


def _score_to_semaforo(score: pd.Series) -> pd.Series:
    numeric_score = pd.to_numeric(score, errors="coerce").fillna(0)
    values = np.select(
        [numeric_score <= 40, numeric_score <= 75, numeric_score > 75],
        ["Verde", "Amarillo", "Rojo"],
        default="Verde",
    )
    return pd.Series(values, index=score.index)


def build_features(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Construye features reutilizables a partir de dataframes ya cargados por notebooks."""
    siniestros, polizas, asegurados, documentos, proveedores, vehiculos = _prepare_tables(tables)

    if siniestros.empty:
        return pd.DataFrame()

    docs_agg = _aggregate_documents(documentos, siniestros)

    base = siniestros.copy()

    if not polizas.empty and "id_poliza" in polizas.columns:
        pol_cols = [
            c for c in [
                "id_poliza", "ramo", "fecha_inicio", "fecha_fin", "prima", "suma_asegurada", "deducible",
                "canal_venta", "ciudad", "estado_poliza",
            ]
            if c in polizas.columns
        ]
        pol = polizas[pol_cols].drop_duplicates("id_poliza")
        pol = pol.rename(columns={"ramo": "ramo_poliza", "ciudad": "ciudad_poliza"})
        base = base.merge(pol, on="id_poliza", how="left")

    if not asegurados.empty and "id_asegurado" in asegurados.columns:
        aseg_cols = [
            c for c in [
                "id_asegurado", "segmento", "antiguedad_meses", "ciudad", "numero_polizas",
                "reclamos_ultimos_12_meses", "mora_actual", "score_cliente_simulado",
            ]
            if c in asegurados.columns
        ]
        aseg = asegurados[aseg_cols].drop_duplicates("id_asegurado")
        aseg = aseg.rename(columns={"ciudad": "ciudad_asegurado"})
        base = base.merge(aseg, on="id_asegurado", how="left")

    if not vehiculos.empty and {"id_poliza", "id_asegurado"}.issubset(vehiculos.columns):
        veh_cols = [
            c for c in [
                "id_poliza", "id_asegurado", "id_vehiculo", "placa", "chasis", "motor", "marca", "modelo", "anio",
            ]
            if c in vehiculos.columns
        ]
        veh = vehiculos[veh_cols].drop_duplicates(["id_poliza", "id_asegurado"])
        base = base.merge(veh, on=["id_poliza", "id_asegurado"], how="left")

    if not proveedores.empty and "id_proveedor" in proveedores.columns and "id_proveedor" in base.columns:
        prov_cols = [
            c for c in [
                "id_proveedor", "tipo", "ciudad", "reclamos_asociados", "monto_promedio_reclamado",
                "porcentaje_casos_observados", "antiguedad_meses",
            ]
            if c in proveedores.columns
        ]
        prov = proveedores[prov_cols].drop_duplicates("id_proveedor")
        prov = prov.rename(columns={"tipo": "tipo_proveedor", "ciudad": "ciudad_proveedor", "antiguedad_meses": "antiguedad_meses_proveedor"})
        base = base.merge(prov, on="id_proveedor", how="left")

    if not docs_agg.empty:
        base = base.merge(docs_agg, on="id_siniestro", how="left")

    base = _dedupe_columns(base)

    base["docs_inconsistentes"] = _to_bool_series(_series(base, "docs_inconsistentes", False))
    base["docs_faltantes"] = _to_bool_series(_series(base, "docs_faltantes", False))

    base["freq_asegurado_18m"] = count_prev_events(base, "id_asegurado", "fecha_ocurrencia")
    base["freq_vehiculo_18m"] = count_prev_events(base, "id_vehiculo", "fecha_ocurrencia")
    base["freq_conductor_18m"] = count_prev_events(base, "id_asegurado", "fecha_ocurrencia")

    cobertura = _series(base, "cobertura", "").astype(str)
    rc_mask = cobertura.str.contains("responsabilidad|terceros|\brc\b", case=False, regex=True, na=False)
    base["is_rc_only"] = rc_mask
    if "id_vehiculo" in base.columns:
        base["rc_prev_vehiculo_18m"] = count_prev_events(base.loc[rc_mask], "id_vehiculo", "fecha_ocurrencia").reindex(base.index, fill_value=0).astype(int)
    else:
        base["rc_prev_vehiculo_18m"] = 0

    similarity_df = build_narrative_similarity_features(base)
    base["ids_siniestros_similares_top5"] = similarity_df["ids_siniestros_similares_top5"].values
    base["max_similitud_textual"] = similarity_df["max_similitud_textual"].values

    base["score_reclamo_vigencia"] = score_reclamo_vigencia(base)
    base["score_demora_robo"] = score_demora_denuncia_robo(base)
    base["score_freq_asegurado"] = score_frecuencia(base["freq_asegurado_18m"], 8, 4)
    base["score_freq_vehiculo"] = score_frecuencia(base["freq_vehiculo_18m"], 6, 3)
    base["score_freq_conductor"] = score_frecuencia(base["freq_conductor_18m"], 8, 4)
    base["score_rc_only"] = score_rc_only(base["rc_prev_vehiculo_18m"])
    base["score_proveedor"] = score_proveedor(base)
    base["score_docs_incompletos"] = score_docs_incompletos(base)
    base["score_docs_inconsistentes"] = score_docs_inconsistentes(base)
    base["score_dinamica_sospechosa"] = score_dinamica_sospechosa(base)
    base["score_sin_tercero"] = score_evento_sin_tercero(base)
    base["score_reporte_tardio"] = score_reporte_tardio(base)
    base["score_monto_suma_asegurada"] = score_monto_suma_asegurada(base)
    base["score_narrativas_similares"] = score_narrativas_similares(base["max_similitud_textual"])

    base["score_total_reglas"] = base[SCORE_COLUMNS].sum(axis=1).clip(upper=100).astype(int)
    base["semaforo_score"] = _score_to_semaforo(base["score_total_reglas"])

    base["alertas_score_activadas"] = base.apply(
        lambda row: [SCORE_RULE_LABELS[col] for col in SCORE_COLUMNS if pd.to_numeric(row.get(col, 0), errors="coerce") > 0],
        axis=1,
    )
    base["ids_siniestros_similares_top5_json"] = base["ids_siniestros_similares_top5"].apply(lambda ids: json.dumps(ids, ensure_ascii=False))

    return _dedupe_columns(base)
