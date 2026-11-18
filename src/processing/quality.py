"""
Análisis de calidad de datos (Sección VI-D.2 y VII-A.4 del informe):
completitud, validez, consistencia, duplicados, valores extremos,
distribución de clases y posible fuga de información.

Produce un reporte estructurado (dict) que se persiste como JSON en
`reports/` — equivalente, a escala de prototipo, al "informe de calidad"
mencionado en la metodología.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import HIGH_MISSINGNESS_THRESHOLD, REPORTS_DIR, TARGET_COLUMN


def profile_missingness(df: pd.DataFrame) -> dict:
    missing = df.isna().mean().sort_values(ascending=False)
    high_missing = missing[missing > HIGH_MISSINGNESS_THRESHOLD]
    return {
        "proporcion_faltantes_por_columna": missing.round(4).to_dict(),
        "columnas_alta_ausencia": high_missing.round(4).to_dict(),
        "umbral_alta_ausencia": HIGH_MISSINGNESS_THRESHOLD,
    }


def profile_duplicates(df: pd.DataFrame, id_column: str) -> dict:
    exact_dupes = int(df.duplicated(keep="first").sum())
    id_dupes = int(df.duplicated(subset=[id_column], keep="first").sum()) if id_column in df else None
    return {"duplicados_exactos": exact_dupes, "duplicados_por_id": id_dupes}


def profile_implausible_values(df: pd.DataFrame) -> dict:
    """Detecta observaciones clínicamente imposibles (regla de validez)."""
    issues = {}
    if "edad" in df.columns:
        issues["edad_fuera_de_rango_0_120"] = int(((df["edad"] < 0) | (df["edad"] > 120)).sum())
    if "imc" in df.columns:
        issues["imc_fuera_de_rango_10_80"] = int(((df["imc"] < 10) | (df["imc"] > 80)).sum())
    return issues


def profile_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    report = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((series < low) | (series > high)).sum())
        report[col] = {"limite_inferior": round(float(low), 3), "limite_superior": round(float(high), 3), "n_outliers": n_outliers}
    return report


def profile_class_balance(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> dict:
    if target_col not in df.columns:
        return {}
    counts = df[target_col].value_counts(dropna=False)
    props = df[target_col].value_counts(normalize=True, dropna=False)
    return {
        "conteo_clases": counts.to_dict(),
        "proporcion_clases": props.round(4).to_dict(),
    }


def build_quality_report(
    df: pd.DataFrame,
    id_column: str,
    numeric_cols: list[str],
    target_col: str = TARGET_COLUMN,
    stage_label: str = "RAW",
) -> dict:
    report = {
        "etapa": stage_label,
        "n_registros": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "completitud": profile_missingness(df),
        "duplicados": profile_duplicates(df, id_column),
        "validez": profile_implausible_values(df),
        "valores_extremos_iqr": profile_outliers_iqr(df, numeric_cols),
        "balance_clases": profile_class_balance(df, target_col),
    }
    return report


def save_quality_report(report: dict, filename: str) -> Path:
    out_path = REPORTS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return out_path
