"""
Entrenamiento y comparación de modelos (Sección VI-C Fase 3, VI-D.5 y VII-A.5):
regresión logística, Random Forest y XGBoost, con validación cruzada,
manejo de desbalance de clases mediante pesos, y partición de prueba
reservada que SOLO se evalúa una vez al final (Fase 4).

Inspirado en el enfoque de pesos de clase y comparación de modelos del
notebook de referencia (PCA/LDA sobre Breast Cancer Wisconsin) que el
usuario adjuntó como ejemplo, adaptado a un pipeline tabular reproducible
con scikit-learn `Pipeline`/`ColumnTransformer` en lugar de PCA manual.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - fallback si xgboost no está instalado
    _HAS_XGBOOST = False

from config import CV_FOLDS, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE
from src.processing.feature_engineering import build_preprocessor, get_feature_lists


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_train_test_split(analytical_df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> SplitData:
    """Separa predictores/objetivo y realiza la partición entrenamiento/prueba
    estratificada por clase (Fase 3). La partición de prueba se reserva y
    sólo se evalúa una vez, al final (Fase 4), para evitar optimismo
    artificial."""
    numeric, categorical = get_feature_lists(analytical_df)
    feature_cols = numeric + categorical
    X = analytical_df[feature_cols]
    y = analytical_df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    return SplitData(X_train, X_test, y_train, y_test)


def _class_weight_dict(y: pd.Series) -> dict[int, float]:
    """Pesos de clase inversamente proporcionales a la frecuencia, igual que
    en el notebook de referencia (max(counts)/counts)."""
    counts = y.value_counts()
    weights = counts.max() / counts
    return weights.round(4).to_dict()


def build_candidate_models(y_train: pd.Series) -> dict[str, Pipeline]:
    class_weights = _class_weight_dict(y_train)
    numeric, categorical = None, None  # se inyectan al construir el preprocesador externamente

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
        scale_pos_weight = float(neg / max(pos, 1))
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
    cruzada estratificada SOLO sobre el conjunto de entrenamiento, evitando
    fuga de información hacia la partición de prueba."""
    numeric, categorical = get_feature_lists(split.X_train)
    preprocessor = build_preprocessor(numeric, categorical)
    candidate_estimators = build_candidate_models(split.y_train)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results: dict[str, dict] = {}

    for name, estimator in candidate_estimators.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        start = time.time()
        cv_scores = cross_val_score(pipeline, split.X_train, split.y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipeline.fit(split.X_train, split.y_train)
        elapsed = time.time() - start

        results[name] = {
            "pipeline": pipeline,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "training_seconds": elapsed,
            "hyperparameters": {
                k: v for k, v in estimator.get_params().items() if not callable(v)
            },
        }
    return results
