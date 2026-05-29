# AGENT.md

Este archivo explica como usar el agente SQL de siniestros y como integrarlo con una API o app.

## Objetivo

El agente permite hacer preguntas en lenguaje natural sobre la tabla final:

```sql
fraud_ia.siniestros_scored_final
```

La tabla contiene siniestros ya procesados, clasificados, con score final, semaforo y explicabilidad.

## Archivo principal

Modulo importable recomendado:

```python
from scripts.ai_agent.claims_agent import ask_claims_agent, create_claims_sql_agent, SqlAgentConfig
```

Tambien se dejo una copia en `scripts/ai-agent/claims_agent.py` por compatibilidad con la estructura previa, pero para importar desde Python se recomienda `scripts.ai_agent` porque el guion medio no es valido en imports normales.

## Variables de entorno

Crear un archivo `.env` en la raiz del proyecto:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraudia-db
DB_USER=postgres
DB_PASSWORD=change_me

GOOGLE_API_KEY=
GOOGLE_MODEL_ID=gemini-2.5-flash
```

## Modelo sugerido

Por defecto el agente usa:

- `gemini-2.5-flash`

Puedes cambiarlo con `GOOGLE_MODEL_ID` si quieres usar otro modelo compatible de Gemini.

## Uso desde Python

```python
from scripts.ai_agent.claims_agent import ask_claims_agent

respuesta = ask_claims_agent("Cuales son los 10 siniestros con mayor score_final?")
print(respuesta)
```

## Uso desde una API

Ejemplo conceptual:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from scripts.ai_agent.claims_agent import ask_claims_agent

app = FastAPI()

class AgentQuestion(BaseModel):
    question: str

@app.post("/agent/sql")
def query_agent(payload: AgentQuestion):
    answer = ask_claims_agent(payload.question)
    return {"answer": answer}
```

## Uso desde CLI

```bash
python scripts/ai_agent/claims_agent.py "Que proveedores concentran mas alertas rojas?"
```

## Prueba en bash

Sin API key de Google AI Studio / Gemini, la prueba solo valida imports y construccion de URI:

```bash
bash tests/test_sql_agent.sh
```

Con token configurado, ejecuta una consulta real:

```bash
bash tests/test_sql_agent.sh run
```

## Seguridad

El agente SQL puede ejecutar consultas generadas por el LLM. Para produccion, usar un usuario PostgreSQL con permisos solo lectura sobre `fraud_ia.siniestros_scored_final`.

La salida del agente es apoyo para revision humana y no constituye acusacion ni decision automatica de fraude.
