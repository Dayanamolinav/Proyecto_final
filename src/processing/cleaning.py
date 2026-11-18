"""
Transformaciones de limpieza: RAW -> CLEAN (Sección VII-A.4 y VII-B.4).

Aplica: corrección/eliminación de observaciones inválidas, eliminación de
duplicados exactos, tratamiento de valores faltantes y remoción de columnas
que producirían fuga de información. Cada paso queda registrado en la
bitácora de trazabilidad con el número de registros/campos afectados.
"""
from __future__ import annotations

import pandas as pd

from config import POST_OUTCOME_LEAKY_COLUMNS, TARGET_COLUMN
from src.data_layer.traceability import TraceEvent, log_event


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(df)
    log_event(
        TraceEvent(
            stage="CLEAN",
            action="eliminacion_duplicados_exactos",
            details={"registros_antes": before, "registros_despues": len(df)},
            rows_affected=removed,
        )
    )
    return df


def remove_clinically_impossible(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina registros con valores clínicamente imposibles y sin forma de
    verificarse (p.ej. edad negativa), según la regla de validez definida en
    el plan de calidad (Sección VI-B)."""
    before = len(df)
    mask_valid = pd.Series(True, index=df.index)
    if "edad" in df.columns:
        mask_valid &= df["edad"].between(0, 120) | df["edad"].isna()
    if "imc" in df.columns:
        mask_valid &= df["imc"].between(10, 80) | df["imc"].isna()
    df = df.loc[mask_valid].reset_index(drop=True)
    removed = before - len(df)
    log_event(
        TraceEvent(
            stage="CLEAN",
            action="eliminacion_valores_clinicamente_imposibles",
            details={"regla": "edad en [0,120] e imc en [10,80]"},
            rows_affected=removed,
        )
    )
    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina variables generadas después del desenlace (fuga de
    información), conservando únicamente predictores disponibles antes del
    resultado que el modelo intenta estimar."""
    cols_present = [c for c in POST_OUTCOME_LEAKY_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_present)
    log_event(
        TraceEvent(
            stage="CLEAN",
            action="eliminacion_columnas_fuga_informacion",
            details={"columnas_eliminadas": cols_present},
            rows_affected=len(cols_present),
        )
    )
    return df


def drop_rows_without_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    removed = before - len(df)
    log_event(
        TraceEvent(
            stage="CLEAN",
            action="eliminacion_registros_sin_variable_objetivo",
            rows_affected=removed,
        )
    )
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Orquesta la secuencia de limpieza RAW -> CLEAN."""
    df = drop_exact_duplicates(df)
    df = remove_clinically_impossible(df)
    df = drop_leaky_columns(df)
    df = drop_rows_without_target(df)
    return df
