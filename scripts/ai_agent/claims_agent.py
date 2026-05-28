from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

DEFAULT_SCHEMA = "fraud_ia"
DEFAULT_TABLE = "siniestros_scored_final"

# Variables esperadas en .env:
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=fraudia-db
# DB_USER=postgres
# DB_PASSWORD=********
# HUGGINGFACEHUB_API_TOKEN=hf_...
# HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
#
# Modelos Hugging Face sugeridos para probar por API:
# - mistralai/Mistral-7B-Instruct-v0.3
# - Qwen/Qwen2.5-7B-Instruct
# - meta-llama/Meta-Llama-3.1-8B-Instruct (puede requerir aceptar terminos)
# - microsoft/Phi-3.5-mini-instruct
# Algunos modelos requieren permisos en Hugging Face o un endpoint inference compatible.


@dataclass(frozen=True)
class SqlAgentConfig:
    schema: str = DEFAULT_SCHEMA
    table: str = DEFAULT_TABLE
    model_id: str | None = None
    temperature: float = 0.1
    max_new_tokens: int = 512
    top_k: int = 10
    verbose: bool = False


def load_env() -> None:
    """Carga .env si python-dotenv esta instalado."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_postgres_uri() -> str:
    """Construye la cadena PostgreSQL desde variables .env.

    Formato base:
    postgresql://username:password@localhost:5432/my_database
    """
    load_env()

    user = quote_plus(os.getenv("DB_USER", "postgres"))
    password = quote_plus(os.getenv("DB_PASSWORD", ""))
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = quote_plus(os.getenv("DB_NAME", "fraudia-db"))

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def create_huggingface_llm(
    model_id: str | None = None,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
):
    """Crea un LLM usando Hugging Face Inference API via LangChain."""
    load_env()

    try:
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    except ImportError as exc:
        raise ImportError(
            "Instala langchain-huggingface para usar el agente SQL con Hugging Face."
        ) from exc

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    selected_model = model_id or os.getenv("HF_MODEL_ID") or "mistralai/Mistral-7B-Instruct-v0.3"

    if not token:
        raise RuntimeError(
            "Falta HUGGINGFACEHUB_API_TOKEN en .env. Crea un token en Hugging Face y agregalo antes de usar el agente."
        )

    llm = HuggingFaceEndpoint(
        repo_id=selected_model,
        task="text-generation",
        huggingfacehub_api_token=token,
        temperature=temperature,
        max_new_tokens=max_new_tokens
        )
    return ChatHuggingFace(llm=llm)


def create_claims_sql_agent(config: SqlAgentConfig | None = None):
    """Crea un SQL Agent de LangChain conectado a la tabla final de siniestros.

    El agente esta pensado para consultas analiticas sobre fraud_ia.siniestros_scored_final.
    Para produccion se recomienda usar un usuario PostgreSQL solo lectura.
    """
    config = config or SqlAgentConfig()

    try:
        from langchain_community.agent_toolkits import create_sql_agent
        from langchain_community.utilities import SQLDatabase
    except ImportError as exc:
        raise ImportError(
            "Instala langchain-community y sqlalchemy para crear el SQL agent."
        ) from exc

    db_uri = build_postgres_uri()
    db = SQLDatabase.from_uri(
        db_uri,
        schema=config.schema,
        include_tables=[config.table],
        sample_rows_in_table_info=3,
    )
    llm = create_huggingface_llm(
        model_id=config.model_id,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
    )

    prefix = f"""
Eres un asistente analitico para siniestros de seguros.
Responde en español, con foco en revision humana y posible riesgo, nunca como acusacion de fraude.
Usa exclusivamente la tabla {config.schema}.{config.table}.
No ejecutes INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE ni CREATE.
Para rankings usa ORDER BY y LIMIT. Para porcentajes usa NULLIF cuando haya divisiones.
Si no hay datos suficientes, dilo claramente.
""".strip()

    return create_sql_agent(
        llm=llm,
        db=db,
        prefix=prefix,
        top_k=config.top_k,
        verbose=config.verbose,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


def ask_claims_agent(question: str, config: SqlAgentConfig | None = None) -> str:
    """Funcion simple para llamar el agente desde una API o app."""
    agent = create_claims_sql_agent(config=config)
    response: Any = agent.invoke({"input": question})
    if isinstance(response, dict):
        return str(response.get("output", response))
    return str(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pregunta al agente SQL de siniestros.")
    parser.add_argument("question", help="Pregunta en lenguaje natural.")
    parser.add_argument("--verbose", action="store_true", help="Muestra pasos intermedios del agente.")
    args = parser.parse_args()

    answer = ask_claims_agent(args.question, config=SqlAgentConfig(verbose=args.verbose))
    print(answer)
