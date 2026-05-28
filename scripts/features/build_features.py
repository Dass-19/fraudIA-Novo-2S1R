from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ingestion.load_data import load_raw_tables


ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def _series(df: pd.DataFrame, column: str, default_value=np.nan) -> pd.Series:
    value = df.get(column)
    if value is None:
        return pd.Series(default_value, index=df.index)
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=df.index)


def score_reclamo_vigencia(df: pd.DataFrame) -> pd.Series:
    start_days = pd.to_numeric(
        _series(df, "dias_desde_inicio_poliza"),
        errors="coerce",
    )
    end_days = pd.to_numeric(
        _series(df, "dias_desde_fin_poliza"),
        errors="coerce",
    ).abs()
    min_days = pd.concat([start_days, end_days], axis=1).min(axis=1)
    score = np.where(min_days <= 10, 8, np.where(min_days <= 30, 4, 0))
    return pd.Series(score, index=df.index)


def score_demora_denuncia_robo(df: pd.DataFrame) -> pd.Series:
    cobertura = _series(df, "cobertura", "")
    is_robo = cobertura.astype(str).str.contains("robo", case=False, na=False)
    days = pd.to_numeric(
        _series(df, "dias_entre_ocurrencia_reporte"),
        errors="coerce",
    )
    score = np.where(days > 2, 8, np.where((days >= 1) & (days <= 2), 4, 0))
    return pd.Series(np.where(is_robo, score, 0), index=df.index)


def score_reporte_tardio(df: pd.DataFrame) -> pd.Series:
    days = pd.to_numeric(
        _series(df, "dias_entre_ocurrencia_reporte"),
        errors="coerce",
    )
    score = np.where(days > 7, 5, np.where((days >= 4) & (days <= 7), 3, 0))
    return pd.Series(score, index=df.index)


def score_frecuencia(
    counts: pd.Series,
    high_pts: int,
    mid_pts: int,
    high_thr: int = 3,
    mid_thr: int = 2,
) -> pd.Series:
    score = np.where(counts >= high_thr, high_pts, 0)
    score = np.where(counts == mid_thr, mid_pts, score)
    return pd.Series(score, index=counts.index)


def score_rc_only(df: pd.DataFrame, prior_counts: pd.Series) -> pd.Series:
    score = np.where(prior_counts > 2, 6, np.where(prior_counts == 1, 3, 0))
    return pd.Series(score, index=df.index)


def score_proveedor(df: pd.DataFrame) -> pd.Series:
    pct_obs = pd.to_numeric(
        _series(df, "porcentaje_casos_observados"),
        errors="coerce",
    )
    reclamos = pd.to_numeric(
        _series(df, "reclamos_asociados"),
        errors="coerce",
    )
    en_lista = pct_obs >= 40
    score = np.where(en_lista, 10, np.where(reclamos > 2, 5, 0))
    return pd.Series(score, index=df.index)


def score_docs_incompletos(df: pd.DataFrame) -> pd.Series:
    docs_completos = _series(df, "documentos_completos", False).fillna(False)
    faltantes = _series(df, "docs_faltantes", False).fillna(False)
    score = np.where(
        (~docs_completos.astype(bool)) | (faltantes.astype(bool)),
        4,
        0,
    )
    return pd.Series(score, index=df.index)


def score_docs_inconsistentes(df: pd.DataFrame) -> pd.Series:
    inconsistentes = _series(df, "docs_inconsistentes", False).fillna(False)
    score = np.where(inconsistentes.astype(bool), 10, 0)
    return pd.Series(score, index=df.index)


def score_dinamica_sospechosa(df: pd.DataFrame) -> pd.Series:
    desc = _series(df, "descripcion", "").astype(str)
    flag = desc.str.contains("atipic|observado", case=False, na=False)
    score = np.where(flag, 6, 0)
    return pd.Series(score, index=df.index)


def score_evento_sin_tercero(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0, index=df.index)


def score_monto_suma_asegurada(df: pd.DataFrame) -> pd.Series:
    monto = pd.to_numeric(_series(df, "monto_reclamado"), errors="coerce")
    suma = pd.to_numeric(_series(df, "suma_asegurada"), errors="coerce")
    mean_est = pd.to_numeric(
        _series(df, "monto_estimado"),
        errors="coerce",
    ).mean()
    ratio = monto / suma
    score = np.where(ratio > 0.95, 5, np.where(monto > 1.5 * mean_est, 4, 0))
    return pd.Series(score, index=df.index)


def count_prev_events(
    df: pd.DataFrame,
    key_col: str,
    date_col: str,
    window_days: int = 548,
) -> pd.Series:
    df = df[[key_col, date_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_idx"] = np.arange(len(df))
    df = df.sort_values([key_col, date_col])

    counts = np.zeros(len(df), dtype=int)
    for key, group in df.groupby(key_col, sort=False):
        if pd.isna(key):
            continue
        dates = group[date_col].to_numpy()
        idxs = group["_idx"].to_numpy()
        for i in range(len(dates)):
            if pd.isna(dates[i]):
                counts[np.where(df["_idx"].to_numpy() == idxs[i])[0][0]] = 0
                continue
            start = dates[i] - np.timedelta64(window_days, "D")
            left = np.searchsorted(dates, start, side="left")
            counts[np.where(df["_idx"].to_numpy() == idxs[i])[0][0]] = i - left

    return pd.Series(counts, index=df["_idx"].values).sort_index()


def build_features(raw_dir: Path | None = None) -> pd.DataFrame:
    tables = load_raw_tables(raw_dir)
    siniestros = tables["siniestros"]
    asegurados = tables["asegurados"]
    polizas = tables["polizas"]
    documentos = tables["documentos"]
    proveedores = tables["beneficiarios_proveedores"]
    vehiculos = tables["vehiculos"]

    for col in [
        "fecha_ocurrencia",
        "fecha_reporte",
        "created_at",
        "updated_at",
    ]:
        if col in siniestros.columns:
            siniestros[col] = pd.to_datetime(siniestros[col], errors="coerce")
    for col in ["fecha_inicio", "fecha_fin"]:
        if col in polizas.columns:
            polizas[col] = pd.to_datetime(polizas[col], errors="coerce")
    for col in ["fecha_emision"]:
        if col in documentos.columns:
            documentos[col] = pd.to_datetime(documentos[col], errors="coerce")

    docs = documentos.copy()
    docs["doc_inconsistente"] = docs.get(
        "inconsistencia_detectada",
        False,
    ).fillna(False)
    docs["doc_inconsistente"] = docs["doc_inconsistente"].astype(bool) | (
        ~docs.get("legible", True).fillna(True).astype(bool)
    )
    docs = docs.merge(
        siniestros[["id_siniestro", "fecha_ocurrencia"]],
        on="id_siniestro",
        how="left",
    )
    docs["factura_previa"] = (
        (docs.get("tipo_documento") == "FACTURA")
        & (docs.get("fecha_emision") < docs.get("fecha_ocurrencia"))
    )
    docs_agg = (
        docs.groupby("id_siniestro")
        .agg(
            docs_inconsistentes=("doc_inconsistente", "max"),
            docs_faltantes=("entregado", lambda s: (~s.fillna(True)).any()),
            factura_previa=("factura_previa", "max"),
        )
        .reset_index()
    )
    docs_agg["docs_inconsistentes"] = docs_agg[
        ["docs_inconsistentes", "factura_previa"]
    ].any(axis=1)
    docs_agg = docs_agg.drop(columns=["factura_previa"])

    base = siniestros.merge(
        polizas,
        on="id_poliza",
        how="left",
        suffixes=("", "_poliza"),
    )
    base = base.merge(
        asegurados,
        on="id_asegurado",
        how="left",
        suffixes=("", "_asegurado"),
    )
    base = base.merge(
        vehiculos,
        on=["id_poliza", "id_asegurado"],
        how="left",
        suffixes=("", "_vehiculo"),
    )
    base = base.merge(
        proveedores,
        on="id_proveedor",
        how="left",
        suffixes=("", "_proveedor"),
    )
    base = base.merge(docs_agg, on="id_siniestro", how="left")

    base["freq_asegurado_18m"] = count_prev_events(
        base,
        "id_asegurado",
        "fecha_ocurrencia",
    )
    base["freq_vehiculo_18m"] = count_prev_events(
        base,
        "id_vehiculo",
        "fecha_ocurrencia",
    )
    base["freq_conductor_18m"] = count_prev_events(
        base,
        "id_asegurado",
        "fecha_ocurrencia",
    )

    rc_mask = base.get("cobertura", "").astype(str).str.contains(
        "responsabilidad|danos a terceros",
        case=False,
        na=False,
    )
    base["is_rc_only"] = rc_mask
    base["rc_prev_vehiculo_18m"] = (
        base.assign(_rc=rc_mask)
        .loc[lambda d: d["_rc"]]
        .pipe(
            lambda d: count_prev_events(
                d,
                "id_vehiculo",
                "fecha_ocurrencia",
            )
        )
        .reindex(base.index, fill_value=0)
    )

    base["score_reclamo_vigencia"] = score_reclamo_vigencia(base)
    base["score_demora_robo"] = score_demora_denuncia_robo(base)
    base["score_freq_asegurado"] = score_frecuencia(
        base["freq_asegurado_18m"],
        8,
        4,
    )
    base["score_freq_vehiculo"] = score_frecuencia(
        base["freq_vehiculo_18m"],
        6,
        3,
    )
    base["score_freq_conductor"] = score_frecuencia(
        base["freq_conductor_18m"],
        8,
        4,
    )
    base["score_rc_only"] = score_rc_only(base, base["rc_prev_vehiculo_18m"])
    base["score_proveedor"] = score_proveedor(base)
    base["score_docs_incompletos"] = score_docs_incompletos(base)
    base["score_docs_inconsistentes"] = score_docs_inconsistentes(base)
    base["score_dinamica_sospechosa"] = score_dinamica_sospechosa(base)
    base["score_sin_tercero"] = score_evento_sin_tercero(base)
    base["score_reporte_tardio"] = score_reporte_tardio(base)
    base["score_monto_suma_asegurada"] = score_monto_suma_asegurada(base)

    base["score_total_reglas"] = base[
        [
            "score_reclamo_vigencia",
            "score_demora_robo",
            "score_freq_asegurado",
            "score_freq_vehiculo",
            "score_freq_conductor",
            "score_rc_only",
            "score_proveedor",
            "score_docs_incompletos",
            "score_docs_inconsistentes",
            "score_dinamica_sospechosa",
            "score_sin_tercero",
            "score_reporte_tardio",
            "score_monto_suma_asegurada",
        ]
    ].sum(axis=1)

    return base


def main(output_path: Path | None = None) -> Path:
    features = build_features()
    out_path = output_path or DATA_PROCESSED / "features_siniestros.csv"
    features.to_csv(out_path, index=False)
    return out_path
