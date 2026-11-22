"""
Capa de ingesta (Sección VII-A.2 y VII-B.2 del informe).

El archivo CSV entra a Snowflake por un Stage [18] con
`scripts/cargar_csv_snowflake.py` (PUT + COPY INTO, ver sql/01_ingesta_stage.sql).
En este prototipo, al no disponer todavía del archivo del hospital,
`load_source_file()` admite cuatro modos (el cuarto, `use_local_synthetic_csv`,
lee la copia local del mismo lote sintético para trabajar sin conexión):

  1. `csv_path`: deshabilitado hasta adaptar y autorizar el esquema del
     archivo real del hospital (ver el ValueError de abajo).
  2. `use_snowflake_csv=True`: lee el lote de 20.000 pacientes SINTÉTICOS
     que ya fue cargado manualmente a Snowflake (tabla
     RAW.PACIENTES_PRUEBA_SINTETICOS), en vez de generar un lote nuevo cada
     vez. Sigue siendo exclusivamente datos de prueba, no reales.
  3. Por defecto: genera un lote sintético nuevo (`generate_synthetic_clinical_data`).
  4. `uploaded_csv`: CSV subido desde el panel de administración para entrenar
     en local; debe traer el mismo esquema que el lote sintético (ver
     `REQUIRED_UPLOAD_COLUMNS`). `csv_path` sigue deshabilitado.

Toda carga registra metadata de origen y momento (trazabilidad), como exige
el informe.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from config import LOCAL_SYNTHETIC_CSV, PATIENT_ID_COLUMN, RAW_DIR, TARGET_COLUMN, SyntheticDataConfig
from src.data_layer.synthetic_data import generate_synthetic_clinical_data
from src.data_layer.traceability import TraceEvent, log_event


def load_from_snowflake_csv_table(
    schema: str = "RAW", table: str = "PACIENTES_PRUEBA_SINTETICOS"
) -> tuple[pd.DataFrame, dict]:
    """Lee el CSV de 20.000 pacientes sintéticos ya cargado en Snowflake
    mediante el Stage (sql/01_ingesta_stage.sql o
    scripts/cargar_csv_snowflake.py), usando la misma conexión configurada
    en snowflake_connection.py."""
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


def load_local_synthetic_csv(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Lee la copia local del mismo lote de 20.000 pacientes SINTÉTICOS que se
    cargó a Snowflake por Stage (sql/01_ingesta_stage.sql). Permite reproducir
    el entrenamiento sin conexión. No sirve para archivos reales: esos siguen
    deshabilitados hasta aprobar su esquema y protección de identificadores."""
    source = Path(path) if path is not None else LOCAL_SYNTHETIC_CSV
    raw_bytes = source.read_bytes()
    df = pd.read_csv(source, encoding="utf-8-sig", dtype={"id_paciente": "string"})
    metadata = {
        "origen": f"csv_sintetico_local:{source.name}",
        "es_sintetico": True,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "n_registros_iniciales": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "columnas": list(df.columns),
    }
    log_event(TraceEvent(stage="INGESTA", action="carga_csv_sintetico_local", details=metadata, rows_affected=len(df)))
    return df, metadata


# Columnas que debe traer un CSV subido desde el panel de administración: las mismas
# del diccionario de datos del lote sintético (documentacion/diccionario_sintetico.md).
REQUIRED_UPLOAD_COLUMNS = [
    "id_paciente", "edad", "sexo", "antecedente_familiar_cancer", "antecedente_personal_benigno",
    "tabaquismo", "imc", "presion_sistolica", "glucosa_mg_dl", "paridad", "ca125_u_ml",
    "resultado_imagen", "fuente", "periodo", TARGET_COLUMN,
]
MIN_UPLOAD_LABELED_ROWS = 100
MIN_UPLOAD_PER_CLASS = 10


def validate_uploaded_frame(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de problemas del CSV subido (vacía si es utilizable)."""
    problems: list[str] = []
    missing = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    if missing:
        problems.append("Faltan columnas: " + ", ".join(missing) + ".")
        return problems
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    invalid = int((target.notna() & ~target.isin([0, 1])).sum() + (df[TARGET_COLUMN].notna() & target.isna()).sum())
    if invalid:
        problems.append(f"La columna {TARGET_COLUMN} solo admite 0, 1 o vacío ({invalid} valores no válidos).")
        return problems
    labeled = target.dropna()
    if len(labeled) < MIN_UPLOAD_LABELED_ROWS:
        problems.append(f"Se necesitan al menos {MIN_UPLOAD_LABELED_ROWS} registros con {TARGET_COLUMN} (hay {len(labeled)}).")
    for label in (0, 1):
        if int((labeled == label).sum()) < MIN_UPLOAD_PER_CLASS:
            problems.append(f"Se necesitan al menos {MIN_UPLOAD_PER_CLASS} registros con {TARGET_COLUMN} = {label}.")
    return problems


def read_uploaded_csv(source) -> pd.DataFrame:
    """Lee un CSV subido (ruta o archivo en memoria) y normaliza los nombres de columna."""
    df = pd.read_csv(source, encoding="utf-8-sig", dtype={PATIENT_ID_COLUMN: "string"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_uploaded_csv(path: str | Path, file_name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Ingiere un CSV subido desde el panel de administración (entrenamiento local).

    Exige el mismo esquema que el lote sintético. Los identificadores directos
    (nombre, documento) no se guardan: `persist_raw` los elimina antes de RAW.
    No se asume que el archivo sea sintético: queda registrado como no verificado."""
    source = Path(path)
    raw_bytes = source.read_bytes()
    df = read_uploaded_csv(source)
    problems = validate_uploaded_frame(df)
    if problems:
        raise ValueError("El CSV no se puede usar: " + " ".join(problems))
    name = file_name or source.name
    metadata = {
        "origen": f"csv_cargado:{name}",
        "es_sintetico": None,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "n_registros_iniciales": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "columnas": list(df.columns),
    }
    log_event(TraceEvent(stage="INGESTA", action="carga_csv_subido_panel_admin", details=metadata, rows_affected=len(df)))
    return df, metadata


def load_source_file(
    csv_path: str | Path | None = None,
    synth_cfg: SyntheticDataConfig | None = None,
    use_snowflake_csv: bool = False,
    use_local_synthetic_csv: bool = False,
    uploaded_csv: str | Path | None = None,
    uploaded_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Ingiere el archivo fuente y devuelve (DataFrame, metadata_de_carga)."""
    if uploaded_csv is not None:
        return load_uploaded_csv(uploaded_csv, uploaded_name)

    if csv_path is not None:
        raise ValueError(
            "La ingesta de CSV está deshabilitada hasta adaptar el esquema y "
            "la protección de identificadores al archivo autorizado del hospital."
        )

    if use_local_synthetic_csv:
        return load_local_synthetic_csv()

    if use_snowflake_csv:
        return load_from_snowflake_csv_table()

    df = generate_synthetic_clinical_data(synth_cfg)
    source_label = "generador_sintetico_pruebas"
    is_synthetic = True

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


def persist_raw(df: pd.DataFrame, filename: str = "clinical_raw.parquet", backend: str | None = None) -> str:
    """Persiste el conjunto tal como fue recibido en la capa RAW.

    `backend="local"` (por defecto) escribe en Parquet local; `backend="snowflake"`
    escribe en el esquema RAW de ONCORISK_DB (ver src/data_layer/storage.py).
    """
    from src.data_layer.storage import pseudonymize_direct_identifiers, write_layer

    raw_df, pseudo_report = pseudonymize_direct_identifiers(df)
    destination = write_layer(raw_df, "raw", filename, backend=backend)

    log_event(
        TraceEvent(
            stage="RAW",
            action="persistencia_raw_seudonimizada",
            details={"destino": destination, **pseudo_report},
            rows_affected=len(raw_df),
        )
    )
    return destination