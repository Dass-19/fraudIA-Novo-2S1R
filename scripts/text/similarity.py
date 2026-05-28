from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_PATH = DATA_PROCESSED / "narrativas_similares.csv"
SPANISH_STOP_WORDS = [
    "a",
    "acá",
    "ahí",
    "al",
    "algo",
    "algunas",
    "algunos",
    "ante",
    "antes",
    "como",
    "con",
    "contra",
    "cual",
    "de",
    "del",
    "desde",
    "e",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "erais",
    "eran",
    "esa",
    "esas",
    "ese",
    "eso",
    "esta",
    "estaba",
    "estaban",
    "este",
    "esto",
    "la",
    "las",
    "lo",
    "los",
    "más",
    "me",
    "mi",
    "mis",
    "no",
    "nos",
    "o",
    "para",
    "pero",
    "por",
    "que",
    "se",
    "sin",
    "sobre",
    "su",
    "sus",
    "también",
    "te",
    "tu",
    "un",
    "una",
    "y",
]


def build_similarity(
    df: pd.DataFrame,
    text_col: str = "descripcion",
    id_col: str = "id_siniestro",
    min_similarity: float = 0.7,
    max_per_record: int = 5,
) -> pd.DataFrame:
    text = df[text_col].fillna("").astype(str).str.strip()
    vectorizer = TfidfVectorizer(
        stop_words=SPANISH_STOP_WORDS,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(text)

    sim_matrix = cosine_similarity(tfidf)
    np.fill_diagonal(sim_matrix, 0.0)

    records = []
    ids = df[id_col].to_numpy()
    for i in range(sim_matrix.shape[0]):
        sims = sim_matrix[i]
        idx_sorted = np.argsort(sims)[::-1]
        count = 0
        for j in idx_sorted:
            if sims[j] < min_similarity:
                break
            records.append(
                {
                    id_col: ids[i],
                    "id_similar": ids[j],
                    "similaridad": float(sims[j]),
                }
            )
            count += 1
            if count >= max_per_record:
                break

    return pd.DataFrame(records)


def main(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    features_path = input_path or (DATA_PROCESSED / "features_siniestros.csv")
    if not features_path.exists():
        raise FileNotFoundError(
            "No se encontro data/processed/features_siniestros.csv. "
            "Ejecuta build_features.py primero."
        )

    df = pd.read_csv(features_path)
    if "descripcion" not in df.columns:
        raise ValueError(
            "La columna 'descripcion' no existe en features_siniestros.csv"
        )

    results = build_similarity(df)
    out_path = output_path or OUTPUT_PATH
    results.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    main()
