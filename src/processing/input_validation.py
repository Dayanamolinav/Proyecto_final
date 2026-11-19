"""Validación de entradas del formulario académico con datos sintéticos.

Estos límites corresponden a la interfaz del prototipo, no a criterios clínicos.
"""
from __future__ import annotations

import math
import numbers

import pandas as pd
from src.processing.derived_variables import derive_variables

NUMERIC_LIMITS = {
    "edad": (18, 100), "imc": (12, 60),
    "presion_sistolica": (80, 220), "glucosa_mg_dl": (50, 350),
    "paridad": (0, 10), "ca125_u_ml": (0, 600),
}
CATEGORIES = {
    "sexo": {"F", "M"}, "tabaquismo": {"nunca", "exfumador", "activo"},
    "antecedente_familiar_cancer": {0, 1}, "antecedente_personal_benigno": {0, 1},
    "resultado_imagen": {"normal", "indeterminado", "sospechoso"},
}


def validate_patient_input(row: dict) -> None:
    """Exige datos completos y valores admitidos por el formulario.

    El imputador del entrenamiento no significa que el formulario deba
    inventar valores ausentes para una consulta individual.
    """
    for name, (lower, upper) in NUMERIC_LIMITS.items():
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"{name}: se requiere un número válido")
        if not lower <= value <= upper:
            raise ValueError(f"{name}: debe estar entre {lower} y {upper}")
    for name in ("edad", "paridad"):
        if not float(row[name]).is_integer():
            raise ValueError(f"{name}: se requiere un número entero")
    for name, choices in CATEGORIES.items():
        value = row.get(name)
        try:
            valid = value in choices
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f"{name}: categoría desconocida o ausente")


def prepare_patient_input(row: dict) -> pd.DataFrame:
    validate_patient_input(row)
    return derive_variables(pd.DataFrame([row]))
