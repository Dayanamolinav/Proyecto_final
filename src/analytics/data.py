"""Fuente reproducible para el análisis exploratorio del CSV sintético distribuido."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.derived_variables import derive_variables

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "synthetic" / "pacientes_sinteticos_20000.csv"
ORIGINAL_COLUMNS = [
    "id_paciente", "edad", "sexo", "antecedente_familiar_cancer",
    "antecedente_personal_benigno", "tabaquismo", "imc", "presion_sistolica",
    "glucosa_mg_dl", "paridad", "ca125_u_ml", "resultado_imagen", "fuente",
    "periodo", "riesgo_oncologico",
]
NUMBER_COLUMNS = ["edad", "imc", "presion_sistolica", "glucosa_mg_dl", "paridad", "ca125_u_ml",
                  "antecedente_familiar_cancer", "antecedente_personal_benigno", "riesgo_oncologico"]


def load_patient_data(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Lee exclusivamente la fuente local; no entrena, imputa ni escribe datos.

    Se conservan filas, nulos y errores de dominio para su revisión. Las
    conversiones numéricas fallidas se contabilizan en los metadatos. La edad
    decimal del CSV se conserva; no se aplica la regla de entrada manual entera.
    """
    source = Path(path) if path is not None else DEFAULT_SOURCE
    if not source.is_file():
        raise FileNotFoundError(f"No se encuentra el CSV de análisis: {source.name}")
    raw_bytes = source.read_bytes()
    frame = pd.read_csv(source, encoding="utf-8-sig", dtype={"id_paciente": "string"})
    missing = sorted(set(ORIGINAL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Faltan columnas de análisis: " + ", ".join(missing))
    conversions = {}
    for column in NUMBER_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        conversions[column] = int((frame[column].notna() & values.isna()).sum())
        frame[column] = values
    for column in ("sexo", "tabaquismo", "resultado_imagen", "fuente", "periodo"):
        frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)
    frame["id_paciente"] = frame["id_paciente"].str.strip().replace("", pd.NA)
    data = derive_variables(frame)
    # pd.cut(...).astype(str) del módulo compartido puede producir texto 'nan'.
    # Para el panel esas ausencias siguen siendo ausencias, nunca categorías reales.
    for column in ("grupo_etario", "imc_categoria"):
        data[column] = data[column].replace({"nan": pd.NA, "<NA>": pd.NA})
    metadata = {
        "label": "Base sintética de OncoRisk",
        "source": "CSV local incluido en el proyecto",
        "file_name": source.name,
        "hash": hashlib.sha256(raw_bytes).hexdigest(),
        "rows": len(frame),
        "patients": int(frame["id_paciente"].nunique()),
        "classification": "Sintética: generador del prototipo, semilla 42",
        "grain": "Una fila por paciente ficticio; no hay consultas repetidas fechadas",
        "original_columns": ORIGINAL_COLUMNS,
        "numeric_coercions": conversions,
        "independent_ids": bool(frame["id_paciente"].notna().all() and not frame["id_paciente"].duplicated().any()),
        "periods": sorted(frame["periodo"].dropna().unique().tolist()),
        "time_scope": "Semestres de procedencia simulados; no fechas de seguimiento",
        "training_scope": "El panel describe este CSV. El modelo individual conserva su entrenamiento previo en otra base sintética.",
    }
    return data, metadata
