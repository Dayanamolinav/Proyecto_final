"""
Prototipo web (Sección VII-A.7 y VII-B.6 del informe):

"El modelo seleccionado será integrado en un prototipo desarrollado con
Streamlit. La aplicación permitirá ingresar las variables requeridas por el
modelo y visualizar la salida probabilística generada, facilitando la
consulta y presentación de los resultados obtenidos."

Esta app:
  1) Carga el modelo seleccionado y persistido por `src/pipeline.py`
     (mismo pipeline de preprocesamiento usado en el entrenamiento).
  2) Recibe variables clínicas válidas mediante un formulario.
  3) Aplica la MISMA canalización usada en el entrenamiento (sin
     reimplementarla) y presenta una estimación probabilística reproducible.
  4) Muestra el aporte de variables (interpretabilidad) y métricas de
     validación del modelo, con las advertencias de alcance que exige la
     Sección VI-D.6 / VII-A.8 (no diagnóstico, no sustituye valoración
     profesional, uso académico/experimental).

Ejecutar con:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODELS_DIR, REPORTS_DIR  # noqa: E402
from src.modeling.explain import explain_model  # noqa: E402
from src.processing.input_validation import prepare_patient_input  # noqa: E402

MODEL_PATH = MODELS_DIR / "modelo_seleccionado.joblib"

st.set_page_config(page_title="Prototipo · Riesgo oncológico", page_icon="🧬", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def render_header():
    st.title("Prototipo académico · Identificación de características asociadas al riesgo oncológico")
    st.caption(
        "Universidad del Norte · Ingeniería de datos e inteligencia artificial aplicada a información clínica"
    )
    st.warning(
        "⚠️ **Alcance académico y experimental.** Esta aplicación es un prototipo construido con "
        "**datos sintéticos de prueba** (el conjunto real del hospital universitario aún no ha sido "
        "entregado). La salida es una **estimación probabilística**, no un diagnóstico, y **no reemplaza "
        "la valoración de un profesional de la salud**, conforme a la Sección VI-E del informe del proyecto.",
        icon="⚠️",
    )


def render_sidebar_model_info(bundle: dict):
    with st.sidebar:
        st.subheader("Modelo seleccionado")
        st.write(f"**Algoritmo:** `{bundle['model_name']}`")
        ev = bundle["evaluation"]["modelo"]
        st.metric("AUROC (prueba reservada)", f"{ev['auroc']:.3f}")
        st.metric("Brier score (prueba reservada)", f"{ev['brier_score']:.3f}")
        c1, c2 = st.columns(2)
        c1.metric("Sensibilidad", f"{ev['sensibilidad']:.2f}")
        c2.metric("Especificidad", f"{ev['especificidad']:.2f}")
        c1.metric("F1", f"{ev['f1']:.2f}")
        c2.metric("PR-AUC", f"{ev['pr_auc']:.2f}")

        null_ev = bundle["evaluation"]["modelo_nulo"]
        st.caption(
            f"Modelo nulo de referencia (prevalencia observada): AUROC {null_ev['auroc']:.3f}, "
            f"Brier {null_ev['brier_score']:.3f}."
        )
        st.caption(
            "Métricas calculadas una única vez sobre el conjunto de prueba reservado, "
            "después de seleccionar el modelo con datos de desarrollo (Fase 4 de la metodología)."
        )

        if (REPORTS_DIR / "resumen_pipeline.json").exists():
            st.caption("Pipeline ejecutado y trazado en `reports/` y `logs/`.")


def patient_input_form() -> pd.DataFrame | None:
    st.subheader("1. Variables clínicas del paciente")
    with st.form("formulario_paciente"):
        col1, col2, col3 = st.columns(3)
        with col1:
            edad = st.number_input("Edad (años)", min_value=18, max_value=100, value=55)
            sexo = st.selectbox("Sexo", ["F", "M"])
            imc = st.number_input("IMC (kg/m²)", min_value=12.0, max_value=60.0, value=26.5, step=0.1)
            tabaquismo = st.selectbox("Tabaquismo", ["nunca", "exfumador", "activo"])
        with col2:
            antecedente_familiar = st.selectbox("Antecedente familiar de cáncer", ["No", "Sí"])
            antecedente_personal = st.selectbox("Antecedente personal benigno relevante", ["No", "Sí"])
            presion_sistolica = st.number_input("Presión sistólica (mmHg)", min_value=80, max_value=220, value=125)
            glucosa = st.number_input("Glucosa (mg/dL)", min_value=50, max_value=350, value=98)
        with col3:
            paridad = st.number_input("Paridad (número de embarazos)", min_value=0, max_value=10, value=1)
            ca125 = st.number_input("CA-125 (U/mL)", min_value=0.0, max_value=600.0, value=18.0, step=0.1)
            resultado_imagen = st.selectbox("Hallazgo de imagen", ["normal", "indeterminado", "sospechoso"])

        submitted = st.form_submit_button("Calcular estimación", use_container_width=True)

    if not submitted:
        return None

    row = {
        "edad": edad,
        "imc": imc,
        "presion_sistolica": presion_sistolica,
        "glucosa_mg_dl": glucosa,
        "paridad": paridad,
        "ca125_u_ml": ca125,
        "sexo": sexo,
        "antecedente_familiar_cancer": 1 if antecedente_familiar == "Sí" else 0,
        "antecedente_personal_benigno": 1 if antecedente_personal == "Sí" else 0,
        "tabaquismo": tabaquismo,
        "resultado_imagen": resultado_imagen,
    }
    try:
        return prepare_patient_input(row)
    except ValueError as exc:
        st.error(f"Entrada inválida: {exc}")
        return None


def render_prediction(bundle: dict, patient_df: pd.DataFrame):
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_columns"]
    X = patient_df[feature_cols]

    proba = float(pipeline.predict_proba(X)[0, 1])

    st.subheader("2. Estimación obtenida")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Probabilidad estimada", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
    with c2:
        st.info(
            "Esta probabilidad corresponde a una **clasificación probabilística de la condición "
            "registrada** en los datos de entrenamiento (no se confirma temporalidad ni horizonte de "
            "predicción con el conjunto sintético actual), consistente con la Sección V-C del informe. "
            "No se han establecido umbrales clínicos para esta salida."
        )

    st.subheader("3. Variables con mayor contribución predictiva en el modelo")
    importance_df = explain_model(pipeline, top_n=10)
    st.dataframe(importance_df, use_container_width=True, hide_index=True)
    st.caption(
        "La contribución expresa aporte a la predicción del modelo, **no una relación causal** "
        "(Marco Conceptual, Sección V-C)."
    )


def main():
    render_header()
    bundle = load_model_bundle()

    if bundle is None:
        st.error(
            "No se encontró un modelo entrenado. Ejecute primero el pipeline:\n\n"
            "```bash\npython scripts/run_pipeline.py\n```"
        )
        st.stop()

    render_sidebar_model_info(bundle)
    patient_df = patient_input_form()

    if patient_df is not None:
        render_prediction(bundle, patient_df)


if __name__ == "__main__":
    main()
