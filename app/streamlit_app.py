"""Portal académico OncoRisk. Presentación institucional y modelo sintético."""
from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

<<<<<<< HEAD
=======
import os
from src.pipeline import run_full_pipeline  # noqa: E402

>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR  # noqa: E402
from src.modeling.explain import explain_model  # noqa: E402
from src.processing.input_validation import prepare_patient_input  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODELS_DIR / "modelo_seleccionado.joblib"
MODEL_NAMES = {"logistic_regression": "Regresión logística", "random_forest": "Random Forest", "xgboost": "XGBoost"}
st.set_page_config(page_title="OncoRisk | Investigación y análisis oncológico", page_icon="◈", layout="wide")


def asset_uri(name: str, mime: str) -> str:
    data = (APP_DIR / "assets" / name).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def markup(content: str):
    st.markdown(content, unsafe_allow_html=True)


def icon(name: str) -> str:
    paths = {
        "profile": '<circle cx="24" cy="16" r="7"/><path d="M10 40v-4a14 14 0 0 1 28 0v4M18 36h12M24 30v12"/>',
        "chart": '<path d="M8 8v32h34M16 30l8-10 8 5 10-15M16 38v-5M26 38v-9M36 38V22"/>',
        "research": '<path d="M19 7h10M22 7v13L10 37a4 4 0 0 0 3 6h22a4 4 0 0 0 3-6L26 20V7M16 30h16M21 35h.1M28 38h.1"/>',
        "shield": '<path d="M24 5 8 12v12c0 10 16 19 16 19s16-9 16-19V12ZM16 24l6 6 11-12"/>',
    }
    return f'<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

<<<<<<< HEAD

=======
def render_admin_panel():
    with st.expander("⚙️ Panel de administración — actualizar el modelo", expanded=False):
        codigo_esperado = os.getenv("ADMIN_PANEL_CODE", "")
        if not codigo_esperado:
            st.caption("No hay código configurado (ADMIN_PANEL_CODE en tu .env) — este panel queda abierto.")
        else:
            ingresado = st.text_input("Código de administrador", type="password", key="admin_code_input")
            if ingresado != codigo_esperado:
                if ingresado:
                    st.error("Código incorrecto.")
                else:
                    st.caption("Ingresa el código de administrador para desbloquear esta sección.")
                return

        st.caption("Vuelve a entrenar el modelo sin usar la terminal.")
        fuente = st.radio(
            "Fuente de datos",
            ["Snowflake (20.000 pacientes)", "Generar lote local pequeño"],
            index=0,
            key="admin_fuente",
        )
        password = ""
        if fuente.startswith("Snowflake"):
            password = st.text_input(
                "Contraseña de Snowflake", type="password", key="admin_password",
                help="No se guarda; solo se usa para esta actualización.",
            )
        if st.button("Entrenar / actualizar modelo", use_container_width=True, key="admin_entrenar"):
            if fuente.startswith("Snowflake") and not password:
                st.error("Ingresa la contraseña de Snowflake.")
                return
            if password:
                os.environ["SNOWFLAKE_PASSWORD"] = password
            with st.spinner("Entrenando modelo... puede tardar uno o dos minutos."):
                try:
                    summary = run_full_pipeline(
                        backend="snowflake",
                        use_snowflake_csv=fuente.startswith("Snowflake"),
                    )
                except Exception as exc:
                    st.error(f"No se pudo entrenar: {exc}")
                    return
            load_model_bundle.clear()
            ev = summary["evaluacion_prueba_reservada"]["modelo"]
            st.success(
                f"Modelo actualizado: {summary['modelo_seleccionado']} "
                f"(AUROC {ev['auroc']:.3f})"
            )
            st.rerun()
            
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312
def render_header():
    logo = asset_uri("brand.svg", "image/svg+xml")
    markup(f'''
    <div id="inicio" class="section-anchor"></div>
    <div class="utility-bar"><div class="utility-inner"><span>Investigación en salud · Universidad del Norte</span><span>Entorno académico &nbsp; / &nbsp; Datos sintéticos</span></div></div>
    <header class="site-header">
      <a class="brand" href="#inicio" target="_self" aria-label="OncoRisk, inicio"><img class="brand-mark" src="{logo}" alt=""/><span><span class="brand-name">Onco<span>Risk</span></span><span class="brand-tagline">INVESTIGACIÓN Y ANÁLISIS</span></span></a>
      <nav class="nav-links" aria-label="Navegación principal"><a href="#inicio" target="_self">INICIO</a><a href="#proyecto" target="_self">EL PROYECTO</a><a href="#herramientas" target="_self">HERRAMIENTAS</a><a href="#modelo" target="_self">EL MODELO</a><a class="nav-cta" href="#explorar" target="_self">Iniciar exploración &nbsp; ↗</a></nav>
    </header>''')


def render_hero():
    photo = asset_uri("equipo-investigacion.png", "image/png")
    markup(f'''
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-content"><span class="hero-kicker">CONOCIMIENTO QUE CONECTA</span><h1 id="hero-title" class="hero-title">La investigación<br>empieza con<br><em>una nueva mirada.</em></h1><p class="hero-copy">Exploramos características asociadas al riesgo oncológico a través de los datos y el aprendizaje automático.</p><div class="hero-actions"><a class="button-link" href="#explorar" target="_self">Explorar el modelo &nbsp; →</a><a class="button-link secondary" href="#proyecto" target="_self">Conoce el proyecto</a></div><div class="hero-label">PROTOTIPO ACADÉMICO &nbsp; · &nbsp; DATOS SINTÉTICOS</div></div>
      <div class="hero-picture"><img src="{photo}" alt="Ilustración generada con IA de un equipo de investigación en salud"/><span class="hero-stat">Datos que inspiran<br><strong>nuevas preguntas.</strong></span><small class="image-credit">Imagen ilustrativa generada con IA</small></div>
    </section>''')


def render_services():
    cards = [
        ("profile", "Exploración individual", "Configura las variables de un caso de ejemplo y consulta la estimación del modelo sintético.", "explorar", "Explorar variables"),
        ("chart", "Resultados comprensibles", "Consulta la probabilidad estimada y conoce el alcance de cada resultado de la demostración.", "resultado", "Ver resultados"),
        ("research", "Investigación transparente", "Revisa las variables del modelo y sus métricas de evaluación en la prueba sintética.", "modelo", "Conocer el modelo"),
    ]
    rendered = "".join(f'<article class="service-card"><span class="service-icon">{icon(i)}</span><h3>{title}</h3><p>{copy}</p><a href="#{anchor}" target="_self">{label} &nbsp; →</a></article>' for i, title, copy, anchor, label in cards)
    markup(f'<section id="herramientas" class="content-width section-anchor"><div class="section-heading"><span class="eyebrow">UN ESPACIO PARA EXPLORAR</span><h2>La información, más cerca de ti</h2><p>Tres formas de acercarte al análisis y comprender cada paso.</p></div><div class="service-grid">{rendered}</div></section>')


def render_project():
    markup(f'''
    <section id="proyecto" class="info-band section-anchor"><div class="content-width info-inner"><div class="info-copy"><span class="eyebrow">NUESTRO PROPÓSITO</span><h2>Transformar los datos<br>en conocimiento.</h2><p>OncoRisk es un proyecto académico de ingeniería de datos y aprendizaje automático. Integra preparación de datos, comparación de modelos y una interfaz para explorar sus resultados.</p><p>Esta demostración trabaja con información sintética y permite estudiar el comportamiento del modelo seleccionado.</p><a href="#metodologia" target="_self" class="text-link">Conoce cómo funciona &nbsp; →</a></div><div class="info-art"><div class="purpose-symbol">{icon("shield")}</div><h3>Una exploración responsable</h3><p>Los datos y el modelo son sintéticos. Las estimaciones son experimentales y no deben usarse para diagnóstico ni decisiones clínicas.</p><div class="purpose-tags"><span>Datos sintéticos</span><span>Validación interna</span><span>Uso académico</span></div></div></div></section>''')


def patient_input_form() -> pd.DataFrame | None:
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
            st.caption("Variables de ejemplo para la demostración sintética.")
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
    st.caption("Corresponde a los valores enviados en el último cálculo. Para actualizarlo, vuelve a pulsar Calcular estimación.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.metric("Probabilidad estimada", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
    with right:
        markup('<div class="panel result-card"><h3>¿Cómo leer este resultado?</h3><p>Es una probabilidad de clasificación de una condición simulada. No representa el riesgo futuro de una persona y no se han establecido umbrales clínicos para esta salida.</p></div>')
    with st.expander("Consultar los valores utilizados en el cálculo"):
        st.dataframe(patient_df, use_container_width=True, hide_index=True)


def render_interpretation(bundle: dict):
    name = html.escape(MODEL_NAMES.get(bundle["model_name"], bundle["model_name"]))
    markup(f'<div class="model-note"><span class="eyebrow">MODELO SELECCIONADO</span><h3>{name}</h3><p>Seleccionado entre regresión logística, Random Forest y XGBoost mediante validación cruzada agrupada por paciente.</p></div>')
    with st.expander("Variables con mayor contribución global", expanded=False):
        st.write("Describen el comportamiento general del modelo. No explican por sí solas un resultado individual ni indican causalidad.")
        explanation = explain_model(bundle["pipeline"], top_n=10)
        value_column = explanation.columns[1]
        st.vega_lite_chart(explanation, {
            "mark": {"type": "bar", "cornerRadiusEnd": 4, "color": "#168eb5"},
            "encoding": {
                "y": {"field": "variable", "type": "nominal", "sort": None, "title": None},
                "x": {"field": value_column, "type": "quantitative", "title": value_column},
                "tooltip": [{"field": "variable", "type": "nominal"}, {"field": value_column, "type": "quantitative"}],
            },
            "height": 300,
            "config": {"view": {"stroke": None}, "axis": {"labelColor": "#42586d", "titleColor": "#42586d"}},
        }, use_container_width=True)
        st.dataframe(explanation, use_container_width=True, hide_index=True)
    with st.expander("Ver desempeño en la prueba sintética", expanded=False):
        ev, ref = bundle["evaluation"]["modelo"], bundle["evaluation"]["modelo_nulo"]
        a, b, c, d = st.columns(4)
        a.metric("AUROC", f"{ev['auroc']:.3f}")
        b.metric("Precisión promedio", f"{ev['pr_auc']:.3f}")
        c.metric("Brier del modelo", f"{ev['brier_score']:.3f}")
        d.metric("Brier de referencia", f"{ref['brier_score']:.3f}")
        st.caption("Evaluación interna con datos sintéticos. Un Brier menor indica mejor desempeño probabilístico; no demuestra utilidad clínica.")


def render_workflow():
    steps = [("01", "Configura", "Ingresa las variables del caso sintético en el formulario."), ("02", "Explora", "Calcula la estimación y consulta los valores utilizados."), ("03", "Comprende", "Revisa las contribuciones globales y las métricas del modelo.")]
    rendered = "".join(f'<article class="workflow-step"><span class="step-number">{n}</span><div><h3>{title}</h3><p>{copy}</p></div></article>' for n, title, copy in steps)
    markup(f'<section id="metodologia" class="content-width section-anchor"><div class="section-heading"><span class="eyebrow">UN PROCESO CLARO</span><h2>Del dato a la exploración</h2><p>Acércate al modelo en tres pasos.</p></div><div class="workflow">{rendered}</div></section>')


def render_footer():
    logo = asset_uri("brand.svg", "image/svg+xml")
    markup(f'''<footer class="site-footer"><div class="footer-grid content-width"><div class="footer-brand"><a class="brand" href="#inicio" target="_self"><img class="brand-mark" src="{logo}" alt=""/><span><span class="brand-name">Onco<span>Risk</span></span><span class="brand-tagline">INVESTIGACIÓN Y ANÁLISIS</span></span></a><p>Un espacio académico para explorar<br>los datos y hacer nuevas preguntas.</p></div><div class="footer-links"><h3>Explora</h3><a href="#proyecto" target="_self">El proyecto</a><a href="#explorar" target="_self">Variables de entrada</a><a href="#resultado" target="_self">Resultados</a></div><div class="footer-links"><h3>Conoce más</h3><a href="#modelo" target="_self">Interpretación del modelo</a><a href="#metodologia" target="_self">Cómo funciona</a><a href="#inicio" target="_self">Volver al inicio ↑</a></div><div class="footer-about"><h3>Proyecto académico</h3><p>Universidad del Norte</p><p>Datos sintéticos · Validación interna<br>Uso experimental</p></div></div><div class="footer-bottom">OncoRisk &nbsp; / &nbsp; Investigación y análisis oncológico &nbsp; / &nbsp; Prototipo académico</div></footer>''')


def main():
    markup(f"<style>{(APP_DIR / 'portal.css').read_text(encoding='utf-8')}</style>")
    bundle = load_model_bundle()
    render_header()
    render_hero()
    render_services()
    render_project()
    with st.container(key="tool-section"):
        markup('<div id="explorar" class="section-anchor"></div><div class="section-heading left"><span class="eyebrow">EXPLORACIÓN INDIVIDUAL</span><h2>Una mirada a las variables</h2><p>Modifica los valores de ejemplo para explorar el comportamiento del modelo. Todos los campos son obligatorios.</p></div>')
        markup('<div class="notice"><strong>Uso académico y experimental.</strong> Los datos y el modelo son sintéticos. El resultado no es un diagnóstico ni debe utilizarse para tomar decisiones clínicas.</div>')
        if bundle is None:
            st.info("La interfaz está lista. Para habilitar las estimaciones, genera el modelo desde la carpeta del proyecto:")
            st.code("python scripts/run_pipeline.py", language="bash")
        else:
            patient_df = patient_input_form()
            if patient_df is not None:
                st.session_state["oncorisk_input"] = patient_df
    with st.container(key="results-section"):
        markup('<div id="resultado" class="section-anchor"></div><div class="section-heading left"><span class="eyebrow">RESULTADOS</span><h2>Comprende tu exploración</h2><p>Consulta la estimación correspondiente al último caso calculado.</p></div>')
        if bundle is not None and "oncorisk_input" in st.session_state:
            render_prediction(bundle, st.session_state["oncorisk_input"])
        else:
            markup(f'<div class="empty-state"><span class="service-icon">{icon("chart")}</span><div><h3>Tu resultado aparecerá aquí</h3><p>Completa las variables y selecciona «Calcular estimación» para comenzar.</p><a href="#explorar" target="_self">Ir al formulario &nbsp; ↑</a></div></div>')
    with st.container(key="interpretation-section"):
        markup('<div id="modelo" class="section-anchor"></div><div class="section-heading left"><span class="eyebrow">INVESTIGACIÓN TRANSPARENTE</span><h2>Detrás de cada estimación</h2><p>Explora cómo se comporta el modelo en esta demostración.</p></div>')
        if bundle is not None:
            render_interpretation(bundle)
        else:
            st.caption("La interpretación y las métricas estarán disponibles después de generar el modelo.")
    render_workflow()
<<<<<<< HEAD
=======
    render_admin_panel()
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312
    render_footer()


if __name__ == "__main__":
    main()
