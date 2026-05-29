from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

try:
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
except ImportError as exc:
    InMemoryChatMessageHistory = None  # type: ignore[assignment]
    RunnableWithMessageHistory = None  # type: ignore[assignment]
    LANGCHAIN_CORE_IMPORT_ERROR = exc
else:
    LANGCHAIN_CORE_IMPORT_ERROR = None

DEFAULT_SCHEMA = "fraud_ia"
DEFAULT_TABLE = "siniestros_scored_final"


@dataclass(frozen=True)
class SqlAgentConfig:
    schema: str = DEFAULT_SCHEMA
    table: str = DEFAULT_TABLE
    model_id: str | None = None
    temperature: float = 0.1
    max_new_tokens: int = 724
    top_k: int = 10
    verbose: bool = False


session_store: dict[str, Any] = {}


def get_session_history(session_id: str) -> Any:
    """Recupera o crea el historial en memoria para una sesion de chat."""
    if InMemoryChatMessageHistory is None:
        raise ImportError(
            "Instala langchain-core para usar memoria conversacional en el agente SQL."
        ) from LANGCHAIN_CORE_IMPORT_ERROR
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


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

    return f"postgresql://{user}:{password}@{host}:{port}/{database}?sslmode=require"



def create_google_gemini_llm(
    model_id: str | None = None,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
):
    """Crea un chat model de Gemini usando Google AI Studio via LangChain."""
    load_env()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "Instala langchain-google-genai para usar el agente SQL con Google AI Studio/Gemini."
        ) from exc

    token = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    selected_model = (
        model_id
        or os.getenv("GOOGLE_MODEL_ID")
        or os.getenv("GEMINI_MODEL_ID")
        or "gemini-2.5-flash"
    )

    if not token:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY en .env. Crea una API key en Google AI Studio y agregala antes de usar el agente."
        )

    return ChatGoogleGenerativeAI(
        model=selected_model,
        api_key=token,
        temperature=temperature,
        max_tokens=max_new_tokens,
        max_retries=2,
    )


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
    llm = create_google_gemini_llm(
        model_id=config.model_id,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
    )

    prefix = f"""
Eres un asistente analitico para siniestros de seguros.
Tu respuesta DEBE estar formateada exclusivamente en HTML semántico (sin etiquetas <html> ni <body>, sin Markdown).
Responde en español, con foco en revision humana y posible riesgo, nunca como acusacion de fraude.
Usa exclusivamente la tabla {config.schema}.{config.table}.
No ejecutes INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE ni CREATE.
Para rankings usa ORDER BY y LIMIT. Para porcentajes usa NULLIF cuando haya divisiones.
Si no hay datos suficientes, dilo claramente.
Usa el historial de chat disponible para resolver referencias como "esos", "los anteriores" o "ese siniestro".
Para preguntas sobre el semaforo siempre usa el campo: semaforo_final.
Si te consultan especificamente por un siniestro, con los campos reglas_criticas_activadas, alertas_score_activadas y explicabilidad puedes complementar tu respuesta.
""".strip()

    suffix = """
Historial de conversacion:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}
""".strip()

    return create_sql_agent(
        llm=llm,
        db=db,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad", "chat_history"],
        top_k=config.top_k,
        verbose=config.verbose,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


def ask_claims_agent(
    question: str,
    session_id: str = "default_session",
    config: SqlAgentConfig | None = None,
) -> str:
    """Funcion simple para llamar el agente desde una API o app, ahora con memoria."""
    if RunnableWithMessageHistory is None:
        raise ImportError(
            "Instala langchain-core para usar memoria conversacional en el agente SQL."
        ) from LANGCHAIN_CORE_IMPORT_ERROR

    agent = create_claims_sql_agent(config=config)

    agent_with_history = RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )

    response: Any = agent_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )

    if isinstance(response, dict):
        return str(response.get("output", response))
    return str(response)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pregunta al agente SQL de siniestros.")
    parser.add_argument("question", help="Pregunta en lenguaje natural.")
    parser.add_argument("--verbose", action="store_true", help="Muestra pasos intermedios.")
    args = parser.parse_args()

    print("Respuesta 1:")
    answer_1 = ask_claims_agent(
        args.question,
        session_id="consola_1",
        config=SqlAgentConfig(verbose=args.verbose),
    )
    print(answer_1)

    print("\nRespuesta 2 (Prueba de contexto):")
    answer_2 = ask_claims_agent(
        "¿Cuáles de esos tienen mayor riesgo?",
        session_id="consola_1",
        config=SqlAgentConfig(verbose=args.verbose),
    )
    print(answer_2)
