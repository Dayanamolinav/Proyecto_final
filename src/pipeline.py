"""
Orquestador del flujo completo, articulado con CRISP-DM y las fases del
informe (Fig. 2): Ingesta -> RAW -> CLEAN -> ANALYTICAL -> EDA -> Modelado ->
Evaluación -> Selección -> Prueba final reservada -> Persistencia del
modelo para el prototipo (Streamlit).

Puede ejecutarse con un archivo CSV real (`csv_path`) cuando el hospital lo
entregue, o sin argumentos para usar el conjunto sintético de prueba.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from config import ANALYTICAL_DIR, CLEAN_DIR, MODELS_DIR, PATIENT_ID_COLUMN, RAW_DIR, REPORTS_DIR, TARGET_COLUMN
from src.data_layer.ingestion import load_source_file, persist_raw
from src.data_layer.storage import read_layer, write_layer
from src.data_layer.synthetic_data import SyntheticDataConfig
from src.modeling.evaluate import evaluate_against_null_model, select_best_model
from src.modeling.explain import explain_model
from src.modeling.registry import ExperimentRecord, log_experiment
from src.modeling.train import fit_and_cross_validate, make_train_test_split
from src.processing.cleaning import run_cleaning_pipeline
from src.processing.feature_engineering import build_analytical_table
from src.processing.quality import build_quality_report, save_quality_report


def run_full_pipeline(
    csv_path: str | Path | None = None,
    synth_cfg: SyntheticDataConfig | None = None,
) -> dict:
    """Ejecuta el flujo de punta a punta y devuelve un resumen con las
    rutas generadas y las métricas del modelo seleccionado."""

    # --- Fase: Ingesta -> RAW -------------------------------------------------
    df_source, ingest_meta = load_source_file(csv_path, synth_cfg)
    persist_raw(df_source, filename="clinical_raw.parquet")
    raw_df = read_layer("raw", "clinical_raw.parquet")

    numeric_cols_for_quality = ["edad", "imc", "presion_sistolica", "glucosa_mg_dl", "paridad", "ca125_u_ml"]
    raw_quality = build_quality_report(raw_df, PATIENT_ID_COLUMN, numeric_cols_for_quality, stage_label="RAW")
    save_quality_report(raw_quality, "calidad_raw.json")

    # --- Fase: RAW -> CLEAN -----------------------------------------------------
    clean_df = run_cleaning_pipeline(raw_df)
    write_layer(clean_df, "clean", "clinical_clean.parquet")

    clean_quality = build_quality_report(clean_df, PATIENT_ID_COLUMN, numeric_cols_for_quality, stage_label="CLEAN")
    save_quality_report(clean_quality, "calidad_clean.json")

    # --- Fase: CLEAN -> ANALYTICAL -----------------------------------------------
    analytical_df = build_analytical_table(clean_df)
    write_layer(analytical_df, "analytical", "clinical_analytical.parquet")

    # --- Fase: Modelado -----------------------------------------------------------
    split = make_train_test_split(analytical_df)
    cv_results = fit_and_cross_validate(split)

    best_name = select_best_model(cv_results)
    best_pipeline = cv_results[best_name]["pipeline"]

    # --- Fase: Evaluación final (conjunto de prueba reservado, una sola vez) ----
    y_proba_test = best_pipeline.predict_proba(split.X_test)[:, 1]
    evaluation = evaluate_against_null_model(split.y_train, split.y_test.to_numpy(), y_proba_test)

    feature_importance = explain_model(best_pipeline)

    # --- Registro de experimentos para cada modelo candidato --------------------
    for name, result in cv_results.items():
        y_proba_this = result["pipeline"].predict_proba(split.X_test)[:, 1]
        test_metrics = evaluate_against_null_model(split.y_train, split.y_test.to_numpy(), y_proba_this)["modelo"]
        record = ExperimentRecord(
            model_name=name,
            random_seed=42,
            cv_folds=5,
            test_size=0.20,
            numeric_features=[c for c in split.X_train.columns if c in ["edad", "imc", "presion_sistolica", "glucosa_mg_dl", "paridad", "ca125_u_ml"]],
            categorical_features=[c for c in split.X_train.columns if c not in ["edad", "imc", "presion_sistolica", "glucosa_mg_dl", "paridad", "ca125_u_ml"]],
            hyperparameters={k: v for k, v in result["hyperparameters"].items() if isinstance(v, (int, float, str, bool, type(None)))},
            cv_metric_mean=result["cv_mean"],
            cv_metric_std=result["cv_std"],
            test_metrics=test_metrics,
            training_seconds=result["training_seconds"],
            n_train=len(split.X_train),
            n_test=len(split.X_test),
        )
        log_experiment(record)

    # --- Persistir modelo seleccionado para el prototipo (app Streamlit) --------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "modelo_seleccionado.joblib"
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "feature_columns": list(split.X_train.columns),
            "evaluation": evaluation,
        },
        model_path,
    )

    summary = {
        "ingesta": ingest_meta,
        "modelo_seleccionado": best_name,
        "resultados_validacion_cruzada": {k: {"cv_mean": v["cv_mean"], "cv_std": v["cv_std"]} for k, v in cv_results.items()},
        "evaluacion_prueba_reservada": evaluation,
        "variables_mas_influyentes": feature_importance.to_dict(orient="records"),
        "ruta_modelo": str(model_path),
        "n_analytical": len(analytical_df),
    }

    with (REPORTS_DIR / "resumen_pipeline.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return summary


if __name__ == "__main__":
    result = run_full_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
