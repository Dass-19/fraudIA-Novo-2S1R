# FraudIA Novo 2S1R

Sistema de detección de fraude para siniestros de seguros. Combina reglas de negocio, features, modelo ML, explicabilidad y un agente SQL para consultas analíticas.

## API FastAPI

La API principal está en:

```bash
scripts/app/main.py
```

Instalación:

```bash
pip install -r requirements.txt
```

Ejecución local:

```bash
uvicorn scripts.app.main:app --reload
```

Swagger/OpenAPI:

```text
http://localhost:8000/docs
```

## Endpoints principales

| Método | Ruta | Descripción |
| --- | --- | --- |
| `GET` | `/health` | Healthcheck de la API |
| `GET` | `/api/v1/fraud/required-files` | Lista los 6 CSV requeridos |
| `POST` | `/api/v1/fraud/score` | Procesa los 6 CSV y devuelve todos los registros sin guardar |
| `POST` | `/api/v1/fraud/ingest` | Procesa los 6 CSV, devuelve todos los registros y hace upsert en PostgreSQL |
| `GET` | `/api/siniestros` | Devuelve los registros persistidos para el dashboard |
| `POST` | `/api/chat` | Consulta el agente SQL en lenguaje natural |
| `POST` | `/api/v1/agent/sql` | Alias de `/api/chat` |

## CSV requeridos por la API

Los endpoints `/api/v1/fraud/score` y `/api/v1/fraud/ingest` reciben `multipart/form-data` con estos campos:

| Campo | Archivo sugerido |
| --- | --- |
| `asegurados` | `asegurados.csv` |
| `beneficiarios_proveedores` | `beneficiarios_proveedores.csv` |
| `documentos` | `documentos.csv` |
| `polizas` | `polizas.csv` |
| `siniestros` | `siniestros.csv` |
| `vehiculos` | `vehiculos.csv` |

Ejemplo con `curl`:

```bash
curl -X POST http://localhost:8000/api/v1/fraud/score \
  -F asegurados=@data/raw/asegurados.csv \
  -F beneficiarios_proveedores=@data/raw/beneficiarios_proveedores.csv \
  -F documentos=@data/raw/documentos.csv \
  -F polizas=@data/raw/polizas.csv \
  -F siniestros=@data/raw/siniestros.csv \
  -F vehiculos=@data/raw/vehiculos.csv
```

La respuesta incluye:

- `processed_rows`
- `persisted`
- `inserted_rows`
- `summary`
- `records` con todos los siniestros procesados

## Variables de entorno

Crear `.env` en la raíz del backend:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraudia-db
DB_USER=postgres
DB_PASSWORD=change_me

GOOGLE_API_KEY=
GOOGLE_MODEL_ID=gemini-2.5-flash

# Opcional. Por defecto queda abierto para desarrollo local.
CORS_ORIGINS=*
```

## Pipeline interno

1. `scripts/ingestion/load_data.py` carga los 6 CSV.
2. `scripts/features/build_features.py` construye features.
3. `scripts/rules/fraud_rules.py` aplica reglas de negocio.
4. `scripts/models/fraud_model.py` clasifica con el modelo ML.
5. `scripts/explainability/explain_score.py` calcula score final y explicación.
6. `scripts/ingestion/load_data.py` normaliza y hace upsert en `fraud_ia.siniestros_scored_final`.
7. `scripts/ai_agent/claims_agent.py` permite preguntas SQL en lenguaje natural sobre la tabla final.

La salida del sistema es apoyo para revisión humana; no constituye acusación ni decisión automática de fraude.
