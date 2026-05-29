# Arquitectura

La solucion esta organizada en una arquitectura hibrida de reglas, Machine Learning, NLP y agente SQL.

## Flujo principal

1. El front o backend entrega 6 archivos CSV: asegurados, beneficiarios/proveedores, documentos, polizas, siniestros y vehiculos.
2. `scripts/ingestion/load_data.py` carga los archivos y arma el diccionario esperado por `build_features`.
3. `scripts/features/build_features.py` genera variables reutilizables: frecuencias, scores por senales, similitud de narrativas y score total de reglas.
4. `scripts/rules/fraud_rules.py` aplica reglas criticas y advertencias de negocio.
5. `scripts/models/fraud_model.py` carga `artifact/final-model/model.pkl` y clasifica cada siniestro con `probabilidad_ml` y `prediccion_ml`.
6. `scripts/explainability/explain_score.py` calcula `score_final`, `semaforo_final` y `explicabilidad`.
7. `load_data.py` hace upsert por `id_siniestro` en `fraud_ia.siniestros_scored_final`.
8. `scripts/ai_agent/claims_agent.py` crea un agente SQL con LangChain para consultar la tabla final en lenguaje natural.

## Tabla destino

La tabla destino es:

```sql
fraud_ia.siniestros_scored_final
```

Se recomienda crearla antes de ejecutar la ingesta usando el archivo SQL de DDL del proyecto.

## Variables de entorno

El proyecto usa `.env` para PostgreSQL y Google AI Studio / Gemini:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraudia-db
DB_USER=postgres
DB_PASSWORD=change_me
GOOGLE_API_KEY=
GOOGLE_MODEL_ID=gemini-2.5-flash
```

## Consideracion de despliegue

Para produccion, el agente SQL debe usar un usuario de base de datos solo lectura. La salida del sistema es una alerta para revision humana, no una acusacion de fraude ni una decision automatica.
