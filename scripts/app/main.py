from __future__ import annotations

import os
import sys
from uuid import uuid4
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.ai_agent.claims_agent import ask_claims_agent
from scripts.ingestion.load_data import (
    DEFAULT_TABLE,
    FINAL_DB_COLUMNS,
    REQUIRED_TABLE_FILES,
    dataframe_to_api_records,
    get_db_connection,
    load_input_tables,
    process_claim_tables,
    upsert_final_dataframe,
)

API_TITLE = "FraudIA API"
API_VERSION = "1.0.0"


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or ["*"]


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "API para procesar siniestros con reglas, modelo ML, explicabilidad "
        "y agente SQL de apoyo a revisión humana."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ApiError(BaseModel):
    detail: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Pregunta en lenguaje natural.")
    session_id: str | None = Field(
        default=None,
        description="Identificador opcional de sesion para mantener memoria conversacional.",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class FraudProcessingResponse(BaseModel):
    ok: bool
    processed_rows: int
    persisted: bool
    inserted_rows: int | None = None
    summary: dict[str, Any]
    records: list[dict[str, Any]]


class RequiredFile(BaseModel):
    field: str
    filename: str


def _validate_csv_upload(field_name: str, upload: UploadFile) -> None:
    filename = upload.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"El archivo '{field_name}' debe ser un CSV. Recibido: {filename or 'sin nombre'}.",
        )


def _build_file_map(
    asegurados: UploadFile,
    beneficiarios_proveedores: UploadFile,
    documentos: UploadFile,
    polizas: UploadFile,
    siniestros: UploadFile,
    vehiculos: UploadFile,
) -> dict[str, Any]:
    uploads = {
        "asegurados": asegurados,
        "beneficiarios_proveedores": beneficiarios_proveedores,
        "documentos": documentos,
        "polizas": polizas,
        "siniestros": siniestros,
        "vehiculos": vehiculos,
    }
    for field_name, upload in uploads.items():
        _validate_csv_upload(field_name, upload)
    return {field_name: upload.file for field_name, upload in uploads.items()}


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    semaforo_counts = {"Rojo": 0, "Amarillo": 0, "Verde": 0, "Sin dato": 0}
    scores: list[float] = []
    monto_total = 0.0

    for record in records:
        semaforo = record.get("semaforo_final") or "Sin dato"
        semaforo_counts[str(semaforo)] = semaforo_counts.get(str(semaforo), 0) + 1

        score_final = record.get("score_final")
        if isinstance(score_final, (int, float)):
            scores.append(float(score_final))

        monto = record.get("monto_reclamado")
        if isinstance(monto, (int, float)):
            monto_total += float(monto)

    return {
        "total": len(records),
        "semaforo_final": semaforo_counts,
        "score_final_promedio": round(sum(scores) / len(scores), 2) if scores else None,
        "monto_reclamado_total": round(monto_total, 2),
    }


def _process_uploaded_files(file_map: dict[str, Any], persist: bool) -> FraudProcessingResponse:
    tables = load_input_tables(file_map=file_map)
    final_df = process_claim_tables(tables)
    inserted_rows = upsert_final_dataframe(final_df) if persist else None
    records = dataframe_to_api_records(final_df)
    return FraudProcessingResponse(
        ok=True,
        processed_rows=len(records),
        persisted=persist,
        inserted_rows=inserted_rows,
        summary=_summary(records),
        records=records,
    )


def _handle_pipeline_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, pd.errors.EmptyDataError):
        raise HTTPException(status_code=400, detail="Uno de los CSV esta vacio o no tiene encabezados validos.") from exc
    if isinstance(exc, (KeyError, ValueError)):
        raise HTTPException(status_code=422, detail=f"Datos de entrada invalidos: {exc}") from exc
    raise HTTPException(status_code=500, detail=f"Error procesando siniestros: {exc}") from exc


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": API_TITLE, "version": API_VERSION}


@app.get("/api/v1/fraud/required-files", response_model=list[RequiredFile], tags=["fraud"])
def required_files() -> list[RequiredFile]:
    return [RequiredFile(field=field, filename=filename) for field, filename in REQUIRED_TABLE_FILES.items()]


@app.post(
    "/api/v1/fraud/score",
    response_model=FraudProcessingResponse,
    responses={400: {"model": ApiError}, 422: {"model": ApiError}, 500: {"model": ApiError}},
    tags=["fraud"],
)
async def score_claims(
    asegurados: UploadFile = File(...),
    beneficiarios_proveedores: UploadFile = File(...),
    documentos: UploadFile = File(...),
    polizas: UploadFile = File(...),
    siniestros: UploadFile = File(...),
    vehiculos: UploadFile = File(...),
) -> FraudProcessingResponse:
    """Procesa los 6 CSV y devuelve todos los siniestros clasificados sin guardar en DB."""
    try:
        file_map = _build_file_map(
            asegurados=asegurados,
            beneficiarios_proveedores=beneficiarios_proveedores,
            documentos=documentos,
            polizas=polizas,
            siniestros=siniestros,
            vehiculos=vehiculos,
        )
        return await run_in_threadpool(_process_uploaded_files, file_map, False)
    except Exception as exc:
        _handle_pipeline_error(exc)
        raise


@app.post(
    "/api/v1/fraud/ingest",
    response_model=FraudProcessingResponse,
    responses={400: {"model": ApiError}, 422: {"model": ApiError}, 500: {"model": ApiError}},
    tags=["fraud"],
)
async def ingest_claims(
    asegurados: UploadFile = File(...),
    beneficiarios_proveedores: UploadFile = File(...),
    documentos: UploadFile = File(...),
    polizas: UploadFile = File(...),
    siniestros: UploadFile = File(...),
    vehiculos: UploadFile = File(...),
) -> FraudProcessingResponse:
    """Procesa los 6 CSV, devuelve todos los registros y hace upsert en PostgreSQL."""
    try:
        file_map = _build_file_map(
            asegurados=asegurados,
            beneficiarios_proveedores=beneficiarios_proveedores,
            documentos=documentos,
            polizas=polizas,
            siniestros=siniestros,
            vehiculos=vehiculos,
        )
        return await run_in_threadpool(_process_uploaded_files, file_map, True)
    except Exception as exc:
        _handle_pipeline_error(exc)
        raise


@app.get("/api/siniestros", response_model=list[dict[str, Any]], tags=["dashboard"])
def list_siniestros(
    semaforo: str | None = Query(default=None, description="Filtro opcional: Rojo, Amarillo o Verde."),
    limit: int | None = Query(default=None, ge=1, le=10000, description="Limite opcional de registros."),
) -> list[dict[str, Any]]:
    """Devuelve los siniestros persistidos para el dashboard."""
    columns = ", ".join(FINAL_DB_COLUMNS)
    query = f"SELECT {columns} FROM {DEFAULT_TABLE}"
    params: list[Any] = []

    if semaforo:
        query += " WHERE semaforo_final = %s"
        params.append(semaforo)

    query += " ORDER BY score_final DESC NULLS LAST, id_siniestro"

    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            names = [desc[0] for desc in cursor.description]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudieron leer siniestros desde PostgreSQL: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()

    df = pd.DataFrame(rows, columns=names)
    return dataframe_to_api_records(df)


@app.post("/api/chat", response_model=ChatResponse, responses={503: {"model": ApiError}}, tags=["agent"])
@app.post("/api/v1/agent/sql", response_model=ChatResponse, responses={503: {"model": ApiError}}, tags=["agent"])
async def query_agent(payload: ChatRequest) -> ChatResponse:
    """Pregunta al agente SQL de siniestros en lenguaje natural."""
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacio.")

    session_id = (payload.session_id or "").strip() or f"chat_{uuid4().hex}"

    try:
        reply = await run_in_threadpool(ask_claims_agent, question, session_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Agente SQL no disponible: {exc}") from exc

    return ChatResponse(reply=reply, session_id=session_id)
