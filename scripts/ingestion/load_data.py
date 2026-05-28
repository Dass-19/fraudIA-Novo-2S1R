from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"

RAW_TABLES = (
    "asegurados",
    "beneficiarios_proveedores",
    "documentos",
    "polizas",
    "siniestros",
    "vehiculos",
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")
    return pd.read_csv(path)


def load_raw_table(name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    base_dir = raw_dir or DATA_RAW
    return load_csv(base_dir / f"{name}.csv")


def load_raw_tables(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return {name: load_raw_table(name, raw_dir=raw_dir) for name in RAW_TABLES}
