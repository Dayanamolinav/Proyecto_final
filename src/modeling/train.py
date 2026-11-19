"""
Entrenamiento y comparación de modelos (Sección VI-C Fase 3, VI-D.5 y VII-A.5):
regresión logística, Random Forest y XGBoost, con validación cruzada
agrupada por paciente, predicciones fuera de pliegue para calibración y
umbral, y partición de prueba reservada que SOLO se evalúa una vez al final
(Fase 4). Los pesos de clase son opcionales (config.USE_CLASS_WEIGHTS).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - fallback si xgboost no está instalado
    _HAS_XGBOOST = False

from config import CV_FOLDS, PATIENT_ID_COLUMN, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE, USE_CLASS_WEIGHTS
from src.processing.feature_engineering import build_preprocessor, get_feature_lists


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series


def make_train_test_split(analytical_df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> SplitData:
    """Separa predictores/objetivo y realiza la partición entrenamiento/prueba
    estratificada por clase (Fase 3). La partición de prueba se reserva y
    sólo se evalúa una vez, al final (Fase 4), para evitar optimismo
    artificial."""
    numeric, categorical = get_feature_lists(analytical_df)
    feature_cols = numeric + categorical
    X = analytical_df[feature_cols]
    y = analytical_df[target_col].astype(int)

    if PATIENT_ID_COLUMN not in analytical_df.columns or analytical_df[PATIENT_ID_COLUMN].isna().any():
        raise ValueError("Se requiere un identificador no nulo para cada paciente.")
    groups = analytical_df[PATIENT_ID_COLUMN]
    patient_targets = pd.DataFrame({"id": groups, "target": y}).groupby("id")["target"]
    if (patient_targets.nunique() != 1).any():
        raise ValueError("Cada paciente debe tener una única clase; revise la unidad de análisis.")
    patient_labels = patient_targets.first()
    if set(patient_labels.unique()) != {0, 1}:
        raise ValueError("La variable objetivo debe contener ambas clases binarias 0 y 1.")
    try:
        ids_train, ids_test = train_test_split(
            patient_labels.index.to_numpy(), test_size=TEST_SIZE, random_state=RANDOM_SEED,
            stratify=patient_labels.values,
        )
    except ValueError as exc:
        raise ValueError("No hay suficientes pacientes por clase para separar desarrollo y prueba.") from exc
    mask_train = groups.isin(ids_train)
    return SplitData(X.loc[mask_train], X.loc[~mask_train], y.loc[mask_train], y.loc[~mask_train], groups.loc[mask_train])


def _class_weight_dict(y: pd.Series) -> dict[int, float]:
    """Pesos de clase inversamente proporcionales a la frecuencia (max(counts)/counts)."""
    counts = y.value_counts()
    weights = counts.max() / counts
    return weights.round(4).to_dict()


def build_candidate_models(y_train: pd.Series, use_class_weights: bool = USE_CLASS_WEIGHTS) -> dict[str, Pipeline]:
    """Modelos candidatos. Con `use_class_weights=False` (opción por defecto,
    ver config.py) se entrenan sin pesos para conservar probabilidades bien
    calibradas; el desbalance se aborda con el umbral elegido en desarrollo."""
    class_weights = _class_weight_dict(y_train) if use_class_weights else None

    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            class_weight=class_weights,
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight=class_weights,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }
    if _HAS_XGBOOST:
        # XGBoost usa scale_pos_weight en lugar de class_weight
        neg, pos = y_train.value_counts().reindex([0, 1]).fillna(0)
        scale_pos_weight = float(neg / max(pos, 1)) if use_class_weights else 1.0
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    return models


def fit_and_cross_validate(
    split: SplitData, scoring: str = "roc_auc"
) -> dict[str, dict]:
    """Ajusta cada modelo candidato dentro de un Pipeline con el
    preprocesador (imputación/codificación/escalado) y evalúa por validación
    cruzada estratificada y agrupada por paciente SOLO sobre el conjunto de
    desarrollo, evitando fuga de información hacia la partición de prueba.

    Además de los AUROC por pliegue, conserva las probabilidades fuera de
    pliegue (`oof_proba`): cada paciente de desarrollo recibe la predicción de
    un modelo que no lo vio. Con ellas se evalúan la calibración (Brier) y se
    elige el umbral sin tocar la prueba reservada."""
    if scoring != "roc_auc":
        raise ValueError("La validación cruzada del prototipo usa AUROC como métrica principal.")
    numeric, categorical = get_feature_lists(split.X_train)

    candidate_estimators = build_candidate_models(split.y_train)

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(cv.split(split.X_train, split.y_train, groups=split.groups_train))
    y_dev = split.y_train.to_numpy()
    results: dict[str, dict] = {}

    for name, estimator in candidate_estimators.items():
        pipeline = Pipeline(steps=[("preprocess", build_preprocessor(numeric, categorical)), ("model", estimator)])
        start = time.time()
        oof_proba = np.full(len(y_dev), np.nan)
        cv_scores = []
        for train_idx, valid_idx in folds:
            fold_model = clone(pipeline).fit(split.X_train.iloc[train_idx], split.y_train.iloc[train_idx])
            oof_proba[valid_idx] = fold_model.predict_proba(split.X_train.iloc[valid_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_dev[valid_idx], oof_proba[valid_idx]))
        pipeline.fit(split.X_train, split.y_train)
        elapsed = time.time() - start

        results[name] = {
            "pipeline": pipeline,
            "cv_scores": [float(s) for s in cv_scores],
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "oof_proba": oof_proba,
            "training_seconds": elapsed,
            "hyperparameters": {
                k: v for k, v in estimator.get_params().items() if not callable(v)
            },
        }
    return results
