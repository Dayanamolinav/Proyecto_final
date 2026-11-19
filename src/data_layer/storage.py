"""
Modelo de almacenamiento en tres capas (Sección VII-A.3 y VII-B.3):

  RAW        -> datos recibidos, sin identificadores personales directos.
  CLEAN      -> datos después de limpieza y transformación.
  ANALYTICAL -> conjunto final preparado para análisis y modelado.

<<<<<<< HEAD
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
=======
Por defecto (backend="local", sin costo) las capas se guardan como Parquet
local (data/raw, data/clean, data/analytical). Con backend="snowflake" (o
DATA_BACKEND=snowflake en el .env), las mismas funciones `write_layer`/
`read_layer` escriben y leen directamente de tu cuenta real de Snowflake
(esquemas RAW/CLEAN/ANALYTICAL de ONCORISK_DB, creados por
sql/00_setup_inicial.sql), usando la conexión ya configurada en
`snowflake_connection.py`. El resto de la canalización (cleaning,
feature_engineering, pipeline) no tiene que cambiar: solo este módulo.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from config import ANALYTICAL_DIR, CLEAN_DIR, DIRECT_IDENTIFIERS, RAW_DIR

load_dotenv()

_LAYER_DIRS = {"raw": RAW_DIR, "clean": CLEAN_DIR, "analytical": ANALYTICAL_DIR}
_LAYER_SCHEMAS = {"raw": "RAW", "clean": "CLEAN", "analytical": "ANALYTICAL"}


def _active_backend(backend: str | None) -> str:
    return backend or os.getenv("DATA_BACKEND", "local")


def _table_name_from_filename(filename: str) -> str:
    """'clinical_raw.parquet' -> 'CLINICAL_RAW' (nombre de tabla en Snowflake)."""
    return Path(filename).stem.upper()
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312


def pseudonymize_direct_identifiers(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
<<<<<<< HEAD
    """Elimina los identificadores directos configurados antes de guardar RAW.

    La identificación de pacientes requiere una estrategia específica sin
    exponer el dato original, en línea con la Sección E ("Antes de cargar los
    datos ... se eliminarán o sustituirán los identificadores personales
    directos").
    """
    # El prototipo no necesita conservar identificadores directos. Eliminarlos
    # evita presentar hashes sin clave como una protección suficiente.
=======
    """Elimina los identificadores directos configurados antes de guardar RAW."""
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312
    columns = DIRECT_IDENTIFIERS if columns is None else columns
    df = df.copy()
    affected = [col for col in columns if col in df.columns]
    df = df.drop(columns=affected)
    return df, {"columnas_eliminadas": affected}

<<<<<<< HEAD
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
=======

def write_layer(df: pd.DataFrame, layer: str, filename: str, backend: str | None = None) -> str:
    if layer not in _LAYER_DIRS:
        raise ValueError(f"Capa desconocida: {layer!r}. Use una de {list(_LAYER_DIRS)}")
    backend = _active_backend(backend)

    if backend == "snowflake":
        from snowflake.connector.pandas_tools import write_pandas

        from src.data_layer.snowflake_connection import snowflake_connection

        schema = _LAYER_SCHEMAS[layer]
        table = _table_name_from_filename(filename)
        df_upload = df.copy()
        df_upload.columns = [c.upper() for c in df_upload.columns]

        with snowflake_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"USE SCHEMA {schema}")
            success, _, n_rows, _ = write_pandas(
                conn,
                df_upload,
                table_name=table,
                schema=schema,
                auto_create_table=True,
                overwrite=True,
            )
        if not success:
            raise RuntimeError(f"La carga a Snowflake ({schema}.{table}) no fue exitosa.")
        return f"{schema}.{table} ({n_rows} filas)"

    out_path = _LAYER_DIRS[layer] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return str(out_path)


def read_layer(layer: str, filename: str, backend: str | None = None) -> pd.DataFrame:
    if layer not in _LAYER_DIRS:
        raise ValueError(f"Capa desconocida: {layer!r}. Use una de {list(_LAYER_DIRS)}")
    backend = _active_backend(backend)

    if backend == "snowflake":
        from src.data_layer.snowflake_connection import snowflake_connection

        schema = _LAYER_SCHEMAS[layer]
        table = _table_name_from_filename(filename)
        with snowflake_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {schema}.{table}")
                df = cur.fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df

>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312
    path = _LAYER_DIRS[layer] / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecute primero el pipeline (scripts/run_pipeline.py)."
        )
<<<<<<< HEAD
    return pd.read_parquet(path)
=======
    return pd.read_parquet(path)
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312
