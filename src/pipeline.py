"""
Orquestador del flujo completo, articulado con CRISP-DM y las fases del
informe (Fig. 2): Ingesta -> RAW -> CLEAN -> ANALYTICAL -> EDA -> Modelado ->
Evaluación -> Selección -> Prueba final reservada -> Persistencia del
modelo para el prototipo (Streamlit).

Puede ejecutarse con el generador sintético local, con el lote de 20.000
pacientes de prueba ya cargado en Snowflake (`use_snowflake_csv=True`), o
con un archivo CSV real (deshabilitado hasta autorizarse).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd

from config import (CV_FOLDS, MODELS_DIR, PATIENT_ID_COLUMN, RANDOM_SEED, REPORTS_DIR, TEST_SIZE,
                    USE_CLASS_WEIGHTS)
from src.data_layer.ingestion import load_source_file, persist_raw
from src.data_layer.storage import read_layer, write_layer
from src.data_layer.synthetic_data import SyntheticDataConfig
from src.data_layer.traceability import TraceEvent, log_event
from src.modeling.evaluate import development_summary, evaluate_against_null_model, select_best_model, select_threshold
from src.modeling.explain import explain_model
from src.modeling.registry import ExperimentRecord, log_experiment
from src.modeling.train import fit_and_cross_validate, make_train_test_split
from src.processing.cleaning import run_cleaning_pipeline
from src.processing.feature_engineering import build_analytical_table, get_feature_lists
from src.processing.quality import build_quality_report, save_quality_report


def run_full_pipeline(
    csv_path: str | Path | None = None,
    synth_cfg: SyntheticDataConfig | None = None,
    backend: str | None = None,
    use_snowflake_csv: bool = True,
    use_local_synthetic_csv: bool = False,
    transform_with_sql: bool = False,
) -> dict:
    """Ejecuta el flujo de punta a punta y devuelve un resumen con las
    rutas generadas y las métricas del modelo seleccionado.

    Con `transform_with_sql=True` (requiere backend="snowflake") las capas
    CLEAN y ANALYTICAL se construyen dentro de Snowflake con sql/02 y sql/03;
    si no, con el código Python equivalente de src/processing."""
    backend = backend or os.getenv("DATA_BACKEND", "local")
    if transform_with_sql and backend != "snowflake":
        raise ValueError("Las transformaciones SQL se ejecutan en Snowflake: use backend='snowflake'.")

    # --- Fase: Ingesta -> RAW -------------------------------------------------
    df_source, ingest_meta = load_source_file(
        csv_path, synth_cfg, use_snowflake_csv=use_snowflake_csv, use_local_synthetic_csv=use_local_synthetic_csv
    )
    persist_raw(df_source, filename="clinical_raw.parquet", backend=backend)
    raw_df = read_layer("raw", "clinical_raw.parquet", backend=backend)

    numeric_cols_for_quality = ["edad", "imc", "presion_sistolica", "glucosa_mg_dl", "paridad", "ca125_u_ml"]
    raw_quality = build_quality_report(raw_df, PATIENT_ID_COLUMN, numeric_cols_for_quality, stage_label="RAW")
    save_quality_report(raw_quality, "calidad_raw.json")

    if transform_with_sql:
        # --- RAW -> CLEAN -> ANALYTICAL dentro de Snowflake (sql/02 y sql/03) ----
        from src.data_layer.snowflake_connection import snowflake_connection
        from src.data_layer.sql_runner import ANALYTICAL_SQL, CLEAN_SQL, run_sql_file

        with snowflake_connection() as conn:
            run_sql_file(conn, CLEAN_SQL)
            acceptance = run_sql_file(conn, ANALYTICAL_SQL)[-1]
        log_event(TraceEvent(stage="CLEAN/ANALYTICAL", action="transformaciones_sql_snowflake",
                             details={"scripts": [CLEAN_SQL.name, ANALYTICAL_SQL.name],
                                      "aceptacion_obligatorios_nulos_y_repetidos": [list(r) for r in acceptance]}))
        clean_df = read_layer("clean", "clinical_clean.parquet", backend=backend)
        clean_quality = build_quality_report(clean_df, PATIENT_ID_COLUMN, numeric_cols_for_quality, stage_label="CLEAN")
        save_quality_report(clean_quality, "calidad_clean.json")
        analytical_df = read_layer("analytical", "clinical_analytical.parquet", backend=backend)
        # Copia local para la app y el panel de evaluación (misma tabla que en Snowflake)
        write_layer(analytical_df, "analytical", "clinical_analytical.parquet", backend="local")
    else:
        # --- Fase: RAW -> CLEAN (Python, equivalente a sql/02) -------------------
        clean_df = run_cleaning_pipeline(raw_df)
        write_layer(clean_df, "clean", "clinical_clean.parquet", backend=backend)

        clean_quality = build_quality_report(clean_df, PATIENT_ID_COLUMN, numeric_cols_for_quality, stage_label="CLEAN")
        save_quality_report(clean_quality, "calidad_clean.json")

        # --- Fase: CLEAN -> ANALYTICAL (Python, equivalente a sql/03) ------------
        analytical_df = build_analytical_table(clean_df)
        write_layer(analytical_df, "analytical", "clinical_analytical.parquet", backend=backend)
        if backend == "snowflake":
            write_layer(analytical_df, "analytical", "clinical_analytical.parquet", backend="local")

    # --- Fase: Modelado -----------------------------------------------------------
    split = make_train_test_split(analytical_df)
    cv_results = fit_and_cross_validate(split)

    best_name = select_best_model(cv_results)
    best_pipeline = cv_results[best_name]["pipeline"]

    # --- Umbral y calibración con datos de DESARROLLO (fuera de pliegue) --------
    threshold_info = select_threshold(split.y_train.to_numpy(), cv_results[best_name]["oof_proba"])
    threshold = threshold_info["umbral"]
    development = {
        name: development_summary(split.y_train, result["oof_proba"], threshold if name == best_name else 0.5)
        for name, result in cv_results.items()
    }
    log_event(TraceEvent(stage="MODELADO", action="seleccion_modelo_y_umbral_en_desarrollo",
                         details={"modelo": best_name, **threshold_info, "usa_pesos_clase": USE_CLASS_WEIGHTS,
                                  "brier_fuera_de_pliegue": development[best_name]["modelo"]["brier_score"],
                                  "brier_nulo_desarrollo": development[best_name]["modelo_nulo"]["brier_score"]},
                         rows_affected=len(split.y_train)))

    # --- Fase: Evaluación final (conjunto de prueba reservado, una sola vez) ----
    y_proba_test = best_pipeline.predict_proba(split.X_test)[:, 1]
    evaluation = evaluate_against_null_model(split.y_train, split.y_test.to_numpy(), y_proba_test, threshold=threshold)

    feature_importance = explain_model(best_pipeline)

    # --- Registro de experimentos para cada modelo candidato --------------------
    numeric_features, categorical_features = get_feature_lists(split.X_train)
    for name, result in cv_results.items():

        test_metrics = evaluation["modelo"] if name == best_name else {}
        record = ExperimentRecord(
            model_name=name,
            random_seed=RANDOM_SEED,
            cv_folds=CV_FOLDS,
            test_size=TEST_SIZE,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            hyperparameters={k: v for k, v in result["hyperparameters"].items() if isinstance(v, (int, float, str, bool, type(None)))},
            cv_metric_mean=result["cv_mean"],
            cv_metric_std=result["cv_std"],
            test_metrics=test_metrics,
            training_seconds=result["training_seconds"],
            n_train=len(split.X_train),
            n_test=len(split.X_test),
            usa_pesos_clase=USE_CLASS_WEIGHTS,
            metricas_fuera_de_pliegue=development[name]["modelo"],
            umbral=threshold_info if name == best_name else {},
            origen_datos=ingest_meta.get("origen", ""),
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
            "threshold": threshold,
            "threshold_info": threshold_info,
            "development": development[best_name],
            "use_class_weights": USE_CLASS_WEIGHTS,
            "data_origin": ingest_meta.get("origen", ""),
        },
        model_path,
    )

    summary = {
        "ingesta": ingest_meta,
        "almacenamiento": backend,
        "transformaciones": "sql_snowflake" if transform_with_sql else "python",
        "modelo_seleccionado": best_name,
        "usa_pesos_clase": USE_CLASS_WEIGHTS,
        "umbral_seleccionado": threshold_info,
        "resultados_validacion_cruzada": {k: {"cv_mean": v["cv_mean"], "cv_std": v["cv_std"]} for k, v in cv_results.items()},
        "desarrollo_fuera_de_pliegue": development,
        "evaluacion_prueba_reservada": evaluation,
        "variables_mas_influyentes": feature_importance.to_dict(orient="records"),
        "ruta_modelo": str(Path("models") / model_path.name),
        "n_analytical": len(analytical_df),
    }

    with (REPORTS_DIR / "resumen_pipeline.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return summary


if __name__ == "__main__":
    result = run_full_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))