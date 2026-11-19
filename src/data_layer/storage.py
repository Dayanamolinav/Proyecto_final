"""
Modelo de almacenamiento en tres capas (Sección VII-A.3 y VII-B.3):

  RAW        -> datos recibidos, sin identificadores personales directos.
  CLEAN      -> datos después de limpieza y transformación.
  ANALYTICAL -> conjunto final preparado para análisis y modelado.

En el prototipo, Snowflake se simula con archivos Parquet locales
particionados por capa (data/raw, data/clean, data/analytical). La interfaz
de funciones (`write_layer` / `read_layer`) es intencionalmente simple para
poder sustituirse más adelante por conectores reales a Snowflake (p.ej.
`snowflake-connector-python` o `snowflake-sqlalchemy`) sin tener que tocar el
resto de la canalización.
"""
from __future__ import annotations


from pathlib import Path

import pandas as pd

from config import ANALYTICAL_DIR, CLEAN_DIR, DIRECT_IDENTIFIERS, RAW_DIR

_LAYER_DIRS = {"raw": RAW_DIR, "clean": CLEAN_DIR, "analytical": ANALYTICAL_DIR}


def pseudonymize_direct_identifiers(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Elimina los identificadores directos configurados antes de guardar RAW.

    La identificación de pacientes requiere una estrategia específica sin
    exponer el dato original, en línea con la Sección E ("Antes de cargar los
    datos ... se eliminarán o sustituirán los identificadores personales
    directos").
    """
    # El prototipo no necesita conservar identificadores directos. Eliminarlos
    # evita presentar hashes sin clave como una protección suficiente.
    columns = DIRECT_IDENTIFIERS if columns is None else columns
    df = df.copy()
    affected = [col for col in columns if col in df.columns]
    df = df.drop(columns=affected)
    return df, {"columnas_eliminadas": affected}

def write_layer(df: pd.DataFrame, layer: str, filename: str) -> Path:
    if layer not in _LAYER_DIRS:
        raise ValueError(f"Capa desconocida: {layer!r}. Use una de {list(_LAYER_DIRS)}")
    out_path = _LAYER_DIRS[layer] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def read_layer(layer: str, filename: str) -> pd.DataFrame:
    if layer not in _LAYER_DIRS:
        raise ValueError(f"Capa desconocida: {layer!r}. Use una de {list(_LAYER_DIRS)}")
    path = _LAYER_DIRS[layer] / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecute primero el pipeline (scripts/run_pipeline.py)."
        )
    return pd.read_parquet(path)
