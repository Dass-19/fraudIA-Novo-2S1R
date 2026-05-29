#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
from scripts.ai_agent.claims_agent import build_postgres_uri
uri = build_postgres_uri()
print("URI PostgreSQL construida:", uri.replace(uri.split('@')[0].split('//')[-1], '***:***'))
PY

RUN_MODE="${1:-}" python - <<'PY'
import os
from scripts.ai_agent.claims_agent import create_claims_sql_agent, SqlAgentConfig

has_google_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
if not has_google_key:
    print("Sin GOOGLE_API_KEY/GEMINI_API_KEY: se omite prueba real del agente.")
    print("Cuando agregues la clave, ejecuta: bash tests/test_sql_agent.sh run")
    raise SystemExit(0)

if os.getenv('RUN_MODE') == 'run':
    agent = create_claims_sql_agent(SqlAgentConfig(verbose=True))
    response = agent.invoke({"input": "Cuantos siniestros hay por semaforo_final?"})
    print(response)
else:
    print("API key encontrada. Para ejecutar una consulta real usa: bash tests/test_sql_agent.sh run")
PY
