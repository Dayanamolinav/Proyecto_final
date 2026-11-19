"""
Evaluación de modelos (Sección V-D "Hipótesis", VI-D.6 y VII-A.6):
discriminación (AUROC, PR-AUC), clasificación bajo un umbral (sensibilidad,
especificidad, VPP, F1) y calibración (Brier score y curva de calibración),
siguiendo el marco de Van Calster et al. [17] citado en el informe.

También implementa el "modelo nulo" (predice siempre la prevalencia
observada) usado como referencia en la Hipótesis de investigación (V-D.1):
el modelo seleccionado debe superar un AUROC de 0.50 y lograr un Brier score
inferior al del modelo nulo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def null_model_probabilities(y_train: pd.Series, n: int) -> np.ndarray:
    """Modelo nulo: asigna a todos los pacientes la prevalencia observada en
    el conjunto de desarrollo (ver Hipótesis, Sección V-D.1)."""
    prevalence = float(y_train.mean())
    return np.full(shape=n, fill_value=prevalence)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensibilidad = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    vpp = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    metrics = {
        "auroc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_proba),
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "valor_predictivo_positivo": vpp,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_proba),
        "umbral_usado": threshold,
        "n_verdaderos_positivos": int(tp),
        "n_falsos_positivos": int(fp),
        "n_verdaderos_negativos": int(tn),
        "n_falsos_negativos": int(fn),
    }
    return {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()}


def compute_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return {"prob_predicha": prob_pred.round(4).tolist(), "prob_observada": prob_true.round(4).tolist()}


def evaluate_against_null_model(y_train: pd.Series, y_test: np.ndarray, y_proba_test: np.ndarray) -> dict:
    """Contrasta el modelo evaluado contra el modelo nulo y contra el azar
    (AUROC = 0.50), tal como exige la Hipótesis de investigación."""
    null_proba = null_model_probabilities(y_train, n=len(y_test))
    model_metrics = compute_metrics(y_test, y_proba_test)
    null_metrics = compute_metrics(y_test, null_proba)

    return {
        "modelo": model_metrics,
        "modelo_nulo": null_metrics,
        "curva_calibracion": compute_calibration_curve(y_test, y_proba_test),
        "supera_azar_auroc_mayor_0_50": bool(model_metrics["auroc"] > 0.50),
        "brier_mejor_que_nulo": bool(model_metrics["brier_score"] < null_metrics["brier_score"]),
    }


def select_best_model(cv_results: dict[str, dict], primary_metric: str = "cv_mean") -> str:
    """Selecciona el modelo con mejor desempeño promedio en validación
    cruzada (Fase 4: 'El modelo final se seleccionará utilizando
    exclusivamente los datos de desarrollo'). El conjunto de prueba se
    evalúa UNA SOLA VEZ, después de esta selección."""
    return max(cv_results.items(), key=lambda kv: kv[1][primary_metric])[0]
