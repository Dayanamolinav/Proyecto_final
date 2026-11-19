"""Visualización de la prueba reservada original, sin ajuste ni selección."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve)

from src.analytics.statistics import wilson_interval
from src.modeling.train import make_train_test_split


def recover_holdout(bundle: dict, path: str | Path) -> pd.DataFrame:
    """Reconstruye semilla/partición previa y coteja las cifras persistidas.

    Si la base o el modelo ya no coinciden no muestra una falsa prueba reservada.
    """
    analytical = pd.read_parquet(path)
    split = make_train_test_split(analytical)
    probability = bundle["pipeline"].predict_proba(split.X_test[bundle["feature_columns"]])[:, 1]
    y = split.y_test.to_numpy(dtype=int)
    expected = bundle["evaluation"]["modelo"]
    cm = confusion_matrix(y, probability >= .5, labels=[0, 1]).ravel()
    stored = [expected[k] for k in ("n_verdaderos_negativos", "n_falsos_positivos", "n_falsos_negativos", "n_verdaderos_positivos")]
    if not np.array_equal(cm, stored) or not np.isclose(roc_auc_score(y, probability), expected["auroc"], atol=.000051):
        raise ValueError("La base analítica y el modelo no reproducen la evaluación persistida. Regenere artefactos coherentes antes de visualizar la prueba reservada.")
    output = analytical.loc[split.X_test.index, ["id_paciente"]].copy()
    output["etiqueta"] = y
    output["probabilidad"] = probability
    return output.reset_index(drop=True)


def diagnostic_results(predictions: pd.DataFrame, threshold: float = .5, resamples: int = 500) -> dict:
    y = predictions.etiqueta.to_numpy(dtype=int)
    p = predictions.probabilidad.to_numpy(dtype=float)
    if not len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Se requieren ambas etiquetas en la muestra de evaluación.")
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metric_functions = {
        "ROC-AUC": lambda a, b: roc_auc_score(a, b),
        "Accuracy (exactitud)": lambda a, b: accuracy_score(a, b >= threshold),
        "Precision (VPP)": lambda a, b: precision_score(a, b >= threshold, zero_division=np.nan),
        "Recall (sensibilidad)": lambda a, b: recall_score(a, b >= threshold, zero_division=np.nan),
        "F1-Score": lambda a, b: f1_score(a, b >= threshold, zero_division=np.nan),
        "Especificidad": lambda a, b: recall_score(a, b >= threshold, pos_label=0, zero_division=np.nan),
        "Precisión promedio (AP)": lambda a, b: average_precision_score(a, b),
        "Brier": lambda a, b: brier_score_loss(a, b),
    }
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(y == k) for k in (0, 1)]
    boot = {k: [] for k in metric_functions}
    for _ in range(resamples):
        index = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        for name, function in metric_functions.items():
            boot[name].append(float(function(y[index], p[index])))
    rows = []
    for name, function in metric_functions.items():
        valid = np.array(boot[name]); valid = valid[np.isfinite(valid)]
        lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (math.nan, math.nan)
        rows.append({"Métrica": name, "Estimación": float(function(y, p)), "IC 95% inferior": lo, "IC 95% superior": hi, "Réplicas válidas": len(valid)})
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    observed, mean_prediction = calibration_curve(y, p, n_bins=8, strategy="quantile")
    matrix = pd.DataFrame([{"Etiqueta observada": f"Etiqueta {actual}", "Clasificación del modelo": f"Etiqueta {prediction}", "Pacientes": int(n)} for actual, prediction, n in [(0, 0, tn), (0, 1, fp), (1, 0, fn), (1, 1, tp)]])
    return {"metrics": pd.DataFrame(rows), "matrix": matrix,
            "roc": pd.DataFrame({"Falsos positivos": fpr, "Sensibilidad": tpr}),
            "pr": pd.DataFrame({"Recall": recall, "Precision": precision}),
            "calibration": pd.DataFrame({"Probabilidad estimada": mean_prediction, "Proporción observada": observed}),
            "n": len(y), "positive": int(y.sum()), "threshold": threshold, "resamples": resamples}
