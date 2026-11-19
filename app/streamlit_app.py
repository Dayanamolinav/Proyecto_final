"""Interfaz académica de demostración con datos sintéticos."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR  # noqa: E402
from src.modeling.explain import explain_model  # noqa: E402
from src.processing.input_validation import prepare_patient_input  # noqa: E402

MODEL_PATH = MODELS_DIR / "modelo_seleccionado.joblib"
st.set_page_config(page_title="OncoRisk | Laboratorio analítico", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink: #122b43; --muted: #596f82; --teal: #087b78; --line: #dce7eb; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stAppViewContainer"] { background: #f5f8f9; color: var(--ink); }
[data-testid="stHeader"] { background: #f5f8f9; }
[data-testid="stSidebar"] { background: #102b3a; }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p { color: #e8f4f3; }
[data-testid="stSidebar"] hr { border-color: #375260; }
.block-container { max-width: 1300px; padding-top: 2.1rem; padding-bottom: 4rem; }
h1,h2,h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; color: var(--ink); }
h1 { font-size: clamp(2.15rem, 4vw, 3.15rem) !important; line-height: 1.1 !important; }
h3 { font-size: 1.15rem !important; }
p, label { color: #304b60; }
.eyebrow { display:inline-block; color:#087b78; background:#e1f3f1; border:1px solid #c9e9e5; border-radius:100px; padding:6px 12px; font-size:.72rem; letter-spacing:.11em; font-weight:800; text-transform:uppercase; margin-bottom: .8rem; }
.hero-copy { max-width: 680px; font-size:1.04rem; line-height:1.7; color:#586f80; margin: 0 0 1.1rem; }
.notice { padding: 14px 18px; background:#fff9e9; border:1px solid #f0dfac; border-left:4px solid #d79f30; border-radius:10px; color:#66531f; line-height:1.5; font-size:.91rem; margin: 1rem 0 1.6rem; }
.sidebar-brand { font-family: 'Manrope', sans-serif; font-size:1.45rem; letter-spacing:-.04em; font-weight:800; color:white; }
.sidebar-brand span { color:#68d3c4; }
.sidebar-caption { color:#a8c3c8; font-size:.78rem; line-height:1.55; margin-top:.35rem; }
.section-kicker { color:#087b78; font-size:.72rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin-top:1.2rem; }
.section-note { color:#647b8c; font-size:.9rem; margin:-.3rem 0 1rem; line-height:1.5; }
.panel { background:#fff; border:1px solid #dce8eb; border-radius:14px; padding:1rem 1.15rem; margin: .5rem 0 1rem; }
.panel p { margin:0; font-size:.9rem; line-height:1.6; }
[data-testid="stForm"] { background:#fff; border:1px solid #dce8eb; border-radius:16px; padding:1.45rem 1.55rem; box-shadow:0 5px 18px rgba(21,48,65,.035); }
[data-testid="stForm"] h3 { margin-bottom:.3rem; }
[data-testid="stMetric"] { background:#fff; border:1px solid #dce8eb; border-radius:12px; padding:1rem; }
[data-testid="stMetricValue"] { color:#0b7773; font-weight:700; }
.stButton > button, [data-testid="stFormSubmitButton"] button { border-radius:9px; min-height:2.8rem; font-weight:700; }
[data-testid="stFormSubmitButton"] button p { color: #ffffff; }
[data-testid="stExpander"] { background:#fff; border:1px solid #dce8eb; border-radius:12px; }
.footer { margin-top:3rem; padding-top:1.25rem; border-top:1px solid #dce7eb; color:#718593; font-size:.78rem; }
@media (max-width: 768px) { .block-container { padding-top: 1rem; } [data-testid="stForm"] { padding:1rem; } }
</style>
"""

@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def sidebar(bundle):
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">◈ &nbsp;Onco<span>Risk</span></div><div class="sidebar-caption">LABORATORIO ANALÍTICO<br>Universidad del Norte</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**Panel de investigación**")
        st.caption("Prototipo experimental · Datos sintéticos")
        st.markdown("**01** &nbsp; Ingreso de variables  \n**02** &nbsp; Resultado del modelo  \n**03** &nbsp; Interpretación global")
        st.divider()
        if bundle:
            names = {"logistic_regression":"Regresión logística", "random_forest":"Random Forest", "xgboost":"XGBoost"}
            st.caption("MODELO SELECCIONADO")
            st.markdown(f"**{names.get(bundle['model_name'], bundle['model_name'])}**")
            st.caption("Validación interna con datos sintéticos")
        st.divider()
        st.caption("Herramienta académica. No usar para diagnóstico ni decisiones clínicas.")


def patient_input_form() -> pd.DataFrame | None:
    st.markdown('<div class="section-kicker">01 / Configuración</div>', unsafe_allow_html=True)
    st.subheader("Variables de entrada")
    st.markdown('<p class="section-note">Modifica los valores de ejemplo para explorar el comportamiento del modelo. Todos los campos son obligatorios.</p>', unsafe_allow_html=True)
    with st.form("formulario_paciente"):
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.markdown("#### Perfil general")
            edad = st.number_input("Edad (años)", min_value=18, max_value=100, value=55)
            sexo = st.selectbox("Sexo", ["F", "M"])
            imc = st.number_input("IMC (kg/m²)", min_value=12.0, max_value=60.0, value=26.5, step=0.1)
            tabaquismo = st.selectbox("Tabaquismo", ["nunca", "exfumador", "activo"])
        with col2:
            st.markdown("#### Antecedentes y hábitos")
            antecedente_familiar = st.selectbox("Antecedente familiar de cáncer", ["No", "Sí"])
            antecedente_personal = st.selectbox("Antecedente personal benigno relevante", ["No", "Sí"])
            presion_sistolica = st.number_input("Presión sistólica (mmHg)", min_value=80, max_value=220, value=125)
            glucosa = st.number_input("Glucosa (mg/dL)", min_value=50, max_value=350, value=98)
        with col3:
            st.markdown("#### Información complementaria")
            paridad = st.number_input("Paridad (número de embarazos)", min_value=0, max_value=10, value=1)
            ca125 = st.number_input("CA-125 (U/mL)", min_value=0.0, max_value=600.0, value=18.0, step=0.1)
            resultado_imagen = st.selectbox("Hallazgo de imagen", ["normal", "indeterminado", "sospechoso"])
            st.caption("Las variables son ejemplos definidos para la demostración sintética.")
        submitted = st.form_submit_button("Calcular estimación →", type="primary", use_container_width=True)
    if not submitted:
        return None
    row = {"edad": edad, "imc": imc, "presion_sistolica": presion_sistolica, "glucosa_mg_dl": glucosa,
           "paridad": paridad, "ca125_u_ml": ca125, "sexo": sexo,
           "antecedente_familiar_cancer": 1 if antecedente_familiar == "Sí" else 0,
           "antecedente_personal_benigno": 1 if antecedente_personal == "Sí" else 0,
           "tabaquismo": tabaquismo, "resultado_imagen": resultado_imagen}
    try:
        return prepare_patient_input(row)
    except ValueError as exc:
        st.session_state.pop("oncorisk_input", None)
        st.error(f"Entrada inválida: {exc}")
        return None


def render_prediction(bundle: dict, patient_df: pd.DataFrame):
    pipeline = bundle["pipeline"]
    proba = float(pipeline.predict_proba(patient_df[bundle["feature_columns"]])[0, 1])
    st.markdown('<div class="section-kicker">02 / Resultado</div>', unsafe_allow_html=True)
    st.subheader("Estimación del modelo")
    st.caption("El resultado corresponde a los valores enviados en el último cálculo. Para actualizarlo, vuelve a pulsar Calcular estimación.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.metric("Probabilidad estimada", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
    with right:
        st.markdown('<div class="panel"><p><strong>¿Cómo leer este resultado?</strong><br>Es una probabilidad de clasificación de una condición simulada. No representa el riesgo futuro de una persona y no se han establecido umbrales clínicos para esta salida.</p></div>', unsafe_allow_html=True)
        st.caption("Generada exclusivamente con el modelo entrenado sobre datos sintéticos.")
    st.markdown('<div class="section-kicker">03 / Exploración</div>', unsafe_allow_html=True)
    st.subheader("Comprende el modelo")
    with st.expander("Variables con mayor contribución global", expanded=False):
        st.write("Estas variables describen el comportamiento general del modelo; no explican por sí solas este resultado individual ni indican causalidad.")
        st.dataframe(explain_model(pipeline, top_n=10), use_container_width=True, hide_index=True)
    with st.expander("Ver desempeño en la prueba sintética", expanded=False):
        ev, ref = bundle["evaluation"]["modelo"], bundle["evaluation"]["modelo_nulo"]
        a, b, c, d = st.columns(4)
        a.metric("AUROC", f"{ev['auroc']:.3f}")
        b.metric("Precisión promedio", f"{ev['pr_auc']:.3f}")
        c.metric("Brier del modelo", f"{ev['brier_score']:.3f}")
        d.metric("Brier de referencia", f"{ref['brier_score']:.3f}")
        st.caption("Evaluación interna con datos sintéticos. Un Brier menor indica mejor desempeño probabilístico; no demuestra utilidad clínica.")


def main():
    st.markdown(STYLE, unsafe_allow_html=True)
    bundle = load_model_bundle()
    sidebar(bundle)
    st.markdown('<div class="eyebrow">Investigación · Prototipo académico</div>', unsafe_allow_html=True)
    st.title("Exploración de características oncológicas")
    st.markdown('<p class="hero-copy">Una vista clara para explorar cómo un modelo de demostración procesa variables y presenta sus resultados.</p>', unsafe_allow_html=True)
    st.markdown('<div class="notice"><strong>Uso académico y experimental.</strong> Los datos y el modelo son sintéticos. El resultado no es un diagnóstico y no debe utilizarse para tomar decisiones clínicas.</div>', unsafe_allow_html=True)
    if bundle is None:
        st.error("No se encontró un modelo entrenado. Ejecuta primero el pipeline sintético:")
        st.code("python scripts/run_pipeline.py", language="bash")
        st.stop()
    patient_df = patient_input_form()
    if patient_df is not None:
        st.session_state["oncorisk_input"] = patient_df
    if "oncorisk_input" in st.session_state:
        render_prediction(bundle, st.session_state["oncorisk_input"])
    else:
        st.markdown('<div class="panel"><p><strong>Tu resultado aparecerá aquí.</strong><br>Completa el formulario y selecciona «Calcular estimación» para iniciar la demostración.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">ONCORISK &nbsp; / &nbsp; Proyecto académico · Universidad del Norte &nbsp; / &nbsp; Entorno de demostración</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
