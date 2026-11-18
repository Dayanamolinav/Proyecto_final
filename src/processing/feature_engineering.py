"""
Transformaciones CLEAN -> ANALYTICAL (Sección VII-A.4 y VII-B.4):
construcción de variables derivadas, codificación de categorías e
imputación de valores faltantes, usando ÚNICAMENTE información disponible
antes del resultado, para alimentar el modelado (scikit-learn Pipelines).

El preprocesamiento (imputación + codificación + escalado) se define como un
`ColumnTransformer` de scikit-learn que se ajusta SOLO sobre el conjunto de
entrenamiento (ver train.py), evitando fuga de información entre partición
de entrenamiento y prueba.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PATIENT_ID_COLUMN, TARGET_COLUMN
from src.data_layer.traceability import TraceEvent, log_event

# Columnas de metadata/procedencia que no son predictoras del riesgo y no
# deben entrar al modelo (se conservan en ANALYTICAL solo para trazabilidad).
NON_FEATURE_COLUMNS = [PATIENT_ID_COLUMN, "nombre", "documento_identidad", "fuente", "periodo", TARGET_COLUMN]

NUMERIC_FEATURES = [
    "edad",
    "imc",
    "presion_sistolica",
    "glucosa_mg_dl",
    "paridad",
    "ca125_u_ml",
]
CATEGORICAL_FEATURES = [
    "sexo",
    "antecedente_familiar_cancer",
    "antecedente_personal_benigno",
    "tabaquismo",
    "resultado_imagen",
]


def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables derivadas simples, pertinentes al problema, a partir
    de información ya disponible (sin usar el desenlace)."""
    df = df.copy()
    if "imc" in df.columns:
        df["imc_categoria"] = pd.cut(
            df["imc"],
            bins=[0, 18.5, 25, 30, 100],
            labels=["bajo_peso", "normal", "sobrepeso", "obesidad"],
        ).astype(str)
    if "edad" in df.columns:
        df["grupo_etario"] = pd.cut(
            df["edad"],
            bins=[0, 40, 55, 70, 120],
            labels=["<40", "40-54", "55-69", "70+"],
        ).astype(str)
    log_event(
        TraceEvent(
            stage="ANALYTICAL",
            action="construccion_variables_derivadas",
            details={"variables_nuevas": ["imc_categoria", "grupo_etario"]},
            rows_affected=len(df),
        )
    )
    return df


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES + ["imc_categoria", "grupo_etario"] if c in df.columns]
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Construye el ColumnTransformer de imputación + codificación + escalado.

    - Numéricas: imputación por mediana (robusta a valores extremos) + escalado estándar.
    - Categóricas: imputación por la categoría más frecuente + one-hot encoding.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def build_analytical_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla ANALYTICAL final a partir de CLEAN: agrega
    variables derivadas y conserva únicamente columnas pertinentes
    (identificador seudonimizado + predictores + metadata + objetivo)."""
    df = add_derived_variables(df_clean)
    numeric, categorical = get_feature_lists(df)
    keep_cols = [PATIENT_ID_COLUMN, "fuente", "periodo"] + numeric + categorical + [TARGET_COLUMN]
    keep_cols = [c for c in keep_cols if c in df.columns]
    analytical_df = df[keep_cols].copy()
    log_event(
        TraceEvent(
            stage="ANALYTICAL",
            action="construccion_tabla_analytical",
            details={"columnas_finales": keep_cols},
            rows_affected=len(analytical_df),
        )
    )
    return analytical_df
