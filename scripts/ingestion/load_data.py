from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO, Mapping, TextIO

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.explainability.explain_score import add_final_score_and_explanation
from scripts.features.build_features import build_features
from scripts.models.fraud_model import classify_claims
from scripts.rules.fraud_rules import integrate_rules_with_features

DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "raw"
DEFAULT_TABLE = "fraud_ia.siniestros_scored_final"

REQUIRED_TABLE_FILES: dict[str, str] = {
    "asegurados": "asegurados.csv",
    "beneficiarios_proveedores": "beneficiarios_proveedores.csv",
    "documentos": "documentos.csv",
    "polizas": "polizas.csv",
    "siniestros": "siniestros.csv",
    "vehiculos": "vehiculos.csv",
}

FINAL_DB_COLUMNS = [
    "id_siniestro",
    "id_poliza",
    "id_asegurado",
    "id_proveedor",
    "id_vehiculo",
    "ramo",
    "cobertura",
    "fecha_ocurrencia",
    "fecha_reporte",
    "monto_reclamado",
    "monto_estimado",
    "monto_pagado",
    "suma_asegurada",
    "estado",
    "sucursal",
    "ciudad_poliza",
    "ciudad_asegurado",
    "ciudad_proveedor",
    "descripcion",
    "documentos_completos",
    "docs_faltantes",
    "docs_inconsistentes",
    "freq_asegurado_18m",
    "freq_vehiculo_18m",
    "freq_conductor_18m",
    "max_similitud_textual",
    "ids_siniestros_similares_top5",
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
    "score_narrativas_similares",
    "score_total_reglas",
    "semaforo_score",
    "probabilidad_ml",
    "prediccion_ml",
    "score_final",
    "semaforo_score_final",
    "semaforo_reglas_criticas",
    "semaforo_final",
    "rf_01_perdida_total_robo",
    "rf_02_adulteracion_doc",
    "rf_03_lista_restrictiva",
    "rf_04_dinamica_imposible",
    "rf_05_borde_vigencia_48h",
    "rf_06_demora_robo_4dias",
    "rf_07_narrativa_clonada",
    "rf_08_score_reglas_alto",
    "rf_09_score_alto_y_ml_riesgo",
    "rf_10_documental_multiple",
    "rf_11_proveedor_recurrente_monto_atipico",
    "rf_12_alta_frecuencia_y_borde_vigencia",
    "reglas_criticas_activadas",
    "alertas_score_activadas",
    "explicabilidad",
    "etiqueta_fraude_simulada",
]

JSONB_COLUMNS = {
    "ids_siniestros_similares_top5",
    "reglas_criticas_activadas",
    "alertas_score_activadas",
}

DATE_COLUMNS = {"fecha_ocurrencia", "fecha_reporte"}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def read_csv_source(source: str | Path | BinaryIO | TextIO | pd.DataFrame) -> pd.DataFrame:
    """Lee un CSV desde path, file-like object o DataFrame ya construido.

    Esta funcion queda lista para ser reemplazada por la capa API cuando el backend
    reciba archivos desde el front. Mientras tanto permite probar con data/raw.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (str, Path)):
        return pd.read_csv(source)

    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:
            pass

    return pd.read_csv(source)


def load_input_tables(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    file_map: Mapping[str, str | Path | BinaryIO | TextIO | pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Carga los 6 archivos requeridos.

    file_map permite pasar archivos recibidos por API:
    {
        "asegurados": UploadFile.file o DataFrame,
        "beneficiarios_proveedores": UploadFile.file o DataFrame,
        "documentos": UploadFile.file o DataFrame,
        "polizas": UploadFile.file o DataFrame,
        "siniestros": UploadFile.file o DataFrame,
        "vehiculos": UploadFile.file o DataFrame,
    }
    """
    tables: dict[str, pd.DataFrame] = {}
    input_dir = Path(input_dir)
    file_map = dict(file_map or {})

    missing: list[str] = []
    for table_name, filename in REQUIRED_TABLE_FILES.items():
        source = file_map.get(table_name, input_dir / filename)
        if isinstance(source, (str, Path)) and not Path(source).exists():
            missing.append(f"{table_name}: {source}")
            continue
        tables[table_name] = read_csv_source(source)

    if missing:
        raise FileNotFoundError("No se encontraron archivos requeridos: " + "; ".join(missing))

    return tables


def process_claim_tables(
    tables: Mapping[str, pd.DataFrame],
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aplica build_features, reglas, modelo ML, score final y explicabilidad."""
    features = build_features(tables)

    # Primera pasada: genera las columnas RF requeridas por el modelo sin usar todavia prediccion ML.
    with_initial_rules = integrate_rules_with_features(features)

    kwargs: dict[str, Any] = {}
    if model_path is not None:
        kwargs["model_path"] = model_path
    if metadata_path is not None:
        kwargs["metadata_path"] = metadata_path

    predicted = classify_claims(with_initial_rules, **kwargs)

    # Segunda pasada: reglas criticas que dependen de probabilidad_ml/prediccion_ml.
    with_final_rules = integrate_rules_with_features(predicted)
    final_df = add_final_score_and_explanation(with_final_rules)
    return final_df


def normalize_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres y tipos para hacer upsert en fraud_ia.siniestros_scored_final."""
    normalized = df.copy()

    rf_rename = {col: col.lower() for col in normalized.columns if col.startswith("RF_")}
    normalized = normalized.rename(columns=rf_rename)

    if "ids_siniestros_similares_top5_json" in normalized.columns:
        normalized["ids_siniestros_similares_top5"] = normalized["ids_siniestros_similares_top5_json"]

    for column in FINAL_DB_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    for column in DATE_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce").dt.date

    for column in JSONB_COLUMNS:
        normalized[column] = normalized[column].apply(_coerce_json_array)

    for column in [c for c in FINAL_DB_COLUMNS if c.startswith("rf_")]:
        normalized[column] = normalized[column].fillna(False).astype(bool)

    return normalized[FINAL_DB_COLUMNS].copy()


def _coerce_json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, str):
        clean = value.strip()
        if clean in {"", "nan", "None", "null"}:
            return []
        try:
            parsed = json.loads(clean)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _pythonize_value(value: Any, is_jsonb: bool = False) -> Any:
    if is_jsonb:
        try:
            from psycopg2.extras import Json
        except ImportError as exc:
            raise ImportError("Instala psycopg2-binary para guardar JSONB en PostgreSQL.") from exc
        return Json(_coerce_json_array(value))

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_db_connection():
    """Crea la conexion PostgreSQL usando variables del archivo .env."""
    _load_dotenv_if_available()

    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError("Instala psycopg2-binary para conectar con PostgreSQL.") from exc

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def upsert_final_dataframe(
    df: pd.DataFrame,
    table_name: str = DEFAULT_TABLE,
    conn: Any | None = None,
) -> int:
    """Hace upsert por id_siniestro en fraud_ia.siniestros_scored_final."""
    try:
        from psycopg2.extras import execute_values
    except ImportError as exc:
        raise ImportError("Instala psycopg2-binary para ejecutar upserts en PostgreSQL.") from exc

    db_df = normalize_for_db(df)
    if db_df.empty:
        return 0

    owns_connection = conn is None
    conn = conn or get_db_connection()

    columns = FINAL_DB_COLUMNS
    insert_columns = ", ".join(columns)
    update_columns = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col != "id_siniestro"])

    query = f"""
        INSERT INTO {table_name} ({insert_columns})
        VALUES %s
        ON CONFLICT (id_siniestro) DO UPDATE SET
            {update_columns};
    """

    values = [
        tuple(_pythonize_value(row[col], is_jsonb=col in JSONB_COLUMNS) for col in columns)
        for _, row in db_df.iterrows()
    ]

    try:
        with conn.cursor() as cursor:
            execute_values(cursor, query, values, page_size=500)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()

    return len(values)


def process_files_and_upsert(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    file_map: Mapping[str, str | Path | BinaryIO | TextIO | pd.DataFrame] | None = None,
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    table_name: str = DEFAULT_TABLE,
) -> pd.DataFrame:
    """Pipeline completo: carga archivos, procesa, clasifica, explica y guarda en PostgreSQL."""
    tables = load_input_tables(input_dir=input_dir, file_map=file_map)
    final_df = process_claim_tables(tables, model_path=model_path, metadata_path=metadata_path)
    upsert_final_dataframe(final_df, table_name=table_name)
    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Procesa siniestros y hace upsert del resultado final en PostgreSQL.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directorio con los 6 CSV requeridos.")
    parser.add_argument("--skip-db", action="store_true", help="Procesa y muestra resumen sin insertar en PostgreSQL.")
    parser.add_argument("--output-csv", default="", help="Ruta opcional para guardar copia local del dataset final.")
    args = parser.parse_args()

    tables = load_input_tables(input_dir=args.input_dir)
    final_df = process_claim_tables(tables)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)

    if not args.skip_db:
        inserted = upsert_final_dataframe(final_df)
        print(f"Upsert completado: {inserted} registros en {DEFAULT_TABLE}.")
    else:
        print(f"Procesados {len(final_df)} registros. No se inserto en DB por --skip-db.")


if __name__ == "__main__":
    main()
