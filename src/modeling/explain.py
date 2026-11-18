"""
Interpretabilidad (Marco Conceptual: "Variable predictora" y Fase 5:
"explicación de variables"). Expresa el aporte de cada variable al modelo
seleccionado como CONTRIBUCIÓN PREDICTIVA, nunca como relación causal, en
línea con lo señalado en la Sección V-B.4 del informe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())


def explain_model(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Devuelve una tabla con la contribución predictiva de cada variable:
    coeficientes para regresión logística, importancias para modelos basados
    en árboles (Random Forest, XGBoost)."""
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    if hasattr(model, "coef_"):
        values = model.coef_.ravel()
        label = "coeficiente (log-odds)"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        label = "importancia_relativa"
    else:  # pragma: no cover
        raise ValueError("El modelo no expone coeficientes ni importancias.")

    df = pd.DataFrame({"variable": feature_names, label: values})
    df["magnitud_absoluta"] = df[label].abs()
    df = df.sort_values("magnitud_absoluta", ascending=False).drop(columns="magnitud_absoluta")
    return df.head(top_n).reset_index(drop=True)
