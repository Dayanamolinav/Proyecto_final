"""
Capa de ingesta (Sección VII-A.2 y VII-B.2 del informe).

En producción, esta etapa recibiría el archivo CSV entregado por el hospital
universitario a través de un Stage de Snowflake [18]. En este prototipo, al
<<<<<<< HEAD
no disponer todavía de ese archivo, `load_source_file()` admite dos modos:

  1. Si se indica `csv_path`, se ingiere ese archivo CSV real (así, cuando el
     hospital entregue los datos, debe adaptarse y autorizarse antes de habilitar su carga).
  2. Si no se indica ninguno, se genera el conjunto sintético de prueba
     (`generate_synthetic_clinical_data`) y se persiste como si fuera el
     archivo recién cargado, dejando constancia en la bitácora de que el
     origen es sintético (no se debe confundir con datos reales de pacientes).
=======
no disponer todavía de ese archivo, `load_source_file()` admite tres modos:

  1. `csv_path`: deshabilitado hasta adaptar y autorizar el esquema del
     archivo real del hospital (ver el ValueError de abajo).
  2. `use_snowflake_csv=True`: lee el lote de 20.000 pacientes SINTÉTICOS
     que ya fue cargado manualmente a Snowflake (tabla
     RAW.PACIENTES_PRUEBA_SINTETICOS), en vez de generar un lote nuevo cada
     vez. Sigue siendo exclusivamente datos de prueba, no reales.
  3. Por defecto: genera un lote sintético nuevo (`generate_synthetic_clinical_data`).
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312

Toda carga registra metadata de origen y momento (trazabilidad), como exige
el informe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RAW_DIR, SyntheticDataConfig
from src.data_layer.synthetic_data import generate_synthetic_clinical_data
from src.data_layer.traceability import TraceEvent, log_event


<<<<<<< HEAD
def load_source_file(csv_path: str | Path | None = None, synth_cfg: SyntheticDataConfig | None = None) -> tuple[pd.DataFrame, dict]:
=======
def load_from_snowflake_csv_table(
    schema: str = "RAW", table: str = "PACIENTES_PRUEBA_SINTETICOS"
) -> tuple[pd.DataFrame, dict]:
    """Lee el CSV de 20.000 pacientes sintéticos ya cargado en Snowflake
    (ver documentacion/subir_csv_a_snowflake.md), usando la misma conexión
    configurada en snowflake_connection.py."""
    from src.data_layer.snowflake_connection import snowflake_connection

    with snowflake_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {schema}.{table}")
            df = cur.fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]

    metadata = {
        "origen": f"snowflake_tabla_prueba:{schema}.{table}",
        "es_sintetico": True,
        "n_registros_iniciales": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "columnas": list(df.columns),
    }
    log_event(
        TraceEvent(
            stage="INGESTA",
            action="carga_tabla_snowflake_prueba",
            details=metadata,
            rows_affected=len(df),
        )
    )
    return df, metadata


def load_source_file(
    csv_path: str | Path | None = None,
    synth_cfg: SyntheticDataConfig | None = None,
    use_snowflake_csv: bool = False,
) -> tuple[pd.DataFrame, dict]:
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312
    """Ingiere el archivo fuente y devuelve (DataFrame, metadata_de_carga)."""
    if csv_path is not None:
        raise ValueError(
            "La ingesta de CSV está deshabilitada hasta adaptar el esquema y "
            "la protección de identificadores al archivo autorizado del hospital."
        )
<<<<<<< HEAD
    else:
        df = generate_synthetic_clinical_data(synth_cfg)
        source_label = "generador_sintetico_pruebas"
        is_synthetic = True
=======

    if use_snowflake_csv:
        return load_from_snowflake_csv_table()

    df = generate_synthetic_clinical_data(synth_cfg)
    source_label = "generador_sintetico_pruebas"
    is_synthetic = True
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312

    metadata = {
        "origen": source_label,
        "es_sintetico": is_synthetic,
        "n_registros_iniciales": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "columnas": list(df.columns),
    }

    log_event(
        TraceEvent(
            stage="INGESTA",
            action="carga_archivo_fuente",
            details=metadata,
            rows_affected=len(df),
        )
    )
    return df, metadata


<<<<<<< HEAD
def persist_raw(df: pd.DataFrame, filename: str = "clinical_raw.parquet") -> Path:
    """Persiste el conjunto tal como fue recibido en la capa RAW.

    RAW conserva los datos como fueron recibidos, UNA VEZ retirados o
    sustituidos los identificadores personales directos (ver Sección VII-B.3:
    "en RAW se conservarán los datos recibidos sin identificadores personales
    directos"). Aquí eliminamos los identificadores configurados
    antes de escribir en RAW.
    """
    from src.data_layer.storage import pseudonymize_direct_identifiers

    raw_df, pseudo_report = pseudonymize_direct_identifiers(df)
    out_path = RAW_DIR / filename
    raw_df.to_parquet(out_path, index=False)
=======
def persist_raw(df: pd.DataFrame, filename: str = "clinical_raw.parquet", backend: str | None = None) -> str:
    """Persiste el conjunto tal como fue recibido en la capa RAW.

    `backend="local"` (por defecto) escribe en Parquet local; `backend="snowflake"`
    escribe en el esquema RAW de ONCORISK_DB (ver src/data_layer/storage.py).
    """
    from src.data_layer.storage import pseudonymize_direct_identifiers, write_layer

    raw_df, pseudo_report = pseudonymize_direct_identifiers(df)
    destination = write_layer(raw_df, "raw", filename, backend=backend)
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312

    log_event(
        TraceEvent(
            stage="RAW",
            action="persistencia_raw_seudonimizada",
<<<<<<< HEAD
            details={"archivo": str(out_path), **pseudo_report},
            rows_affected=len(raw_df),
        )
    )
    return out_path
=======
            details={"destino": destination, **pseudo_report},
            rows_affected=len(raw_df),
        )
    )
    return destination
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312
