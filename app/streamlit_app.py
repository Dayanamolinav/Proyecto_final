"""Portal académico OncoRisk. Portada editorial con videos y modelo sintético."""
from __future__ import annotations

import hmac
import html
import io
import os
import sys
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # ADMIN_PANEL_CODE y credenciales, sin importar desde dónde se lance
from config import MODELS_DIR, TEST_SIZE  # noqa: E402
from src.modeling.explain import explain_model  # noqa: E402
from src.processing.input_validation import prepare_patient_input  # noqa: E402
from src.pipeline import run_full_pipeline  # noqa: E402
from src.data_layer.ingestion import read_uploaded_csv, validate_uploaded_frame  # noqa: E402
from app.statistics_panel import render_statistics_panel, VARIABLE_LABELS, category  # noqa: E402
from app.dashboard_cards import AXIS  # noqa: E402
from app.detail_views import _table, record_table  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODELS_DIR / "modelo_seleccionado.joblib"
MEDIA = "app/static/media"  # servido por Streamlit (server.enableStaticServing)
MODEL_NAMES = {"logistic_regression": "Regresión logística", "random_forest": "Random Forest", "xgboost": "XGBoost"}
st.set_page_config(page_title="OncoRisk | Investigación y análisis oncológico", page_icon=":material/biotech:", layout="wide")


def section(content: str):
    """Inserta una sección de la portada (HTML propio, sin JavaScript)."""
    st.html(f'<div class="or">{content}</div>')


def video(name: str, extra: str = "") -> str:
    return (f'<video {extra} autoplay muted loop playsinline preload="auto" poster="{MEDIA}/{name}_poster.jpg">'
            f'<source src="{MEDIA}/{name}.mp4" type="video/mp4"></video>')


def num(value: float, decimals: int = 0) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# --------------------------------------------------------------------------- portada

def render_header(solid: bool, stats: bool = False):
    home = "?" if (solid or stats) else ""
    section(f'''
    <header class="or-header{' solid' if solid else ''}" id="or-header">
      <a class="logo" href="{home}#inicio" target="_self">OncoRisk</a>
      <nav aria-label="Secciones">
        <a href="{home}#proyecto" target="_self">Proyecto</a><a href="{home}#metodo" target="_self">Método</a>
        <a href="{home}#explorar" target="_self">Explorar</a><a href="?vista=estadisticas" target="_self">Análisis estadístico</a>
      </nav>
    </header>''')


def render_hero():
    section(f'''
    <section class="or-hero" id="inicio">
      {video("v13_celulas", 'id="or-hero-video"')}
      <div class="or-hero-in">
        <h1 class="or-d" aria-label="OncoRisk"><span class="ch">Onco</span><span class="ch" style="animation-delay:.12s">Risk</span></h1>
        <div class="or-hero-meta">
          <div><b>Universidad del Norte</b>Ingeniería de Sistemas</div>
          <div><b>Ingeniería de datos</b>Snowflake · SQL · Python</div>
          <div><b>Aprendizaje automático</b>Tres modelos comparados</div>
          <div><b>Prototipo académico</b>Datos sintéticos</div>
        </div>
      </div>
    </section>''')


def render_statement():
    section('''
    <section class="or-statement" id="proyecto">
      <p id="or-manifiesto">Integramos, limpiamos y analizamos <span class="hl">20.000 registros clínicos</span> para identificar qué
      características se asocian con un mayor <span class="hl">riesgo oncológico</span>, y dejamos rastro de cada paso.</p>
    </section>''')


def render_facts(bundle: dict | None):
    if bundle is not None:
        ev, null = bundle["evaluation"]["modelo"], bundle["evaluation"]["modelo_nulo"]
        n_test = int(ev["n_verdaderos_positivos"] + ev["n_falsos_positivos"] + ev["n_verdaderos_negativos"] + ev["n_falsos_negativos"])
        n_total = round(n_test / TEST_SIZE)
        facts = [
            (num(n_total), f'data-count="{n_total}" data-sep="1"', "pacientes sintéticos en la base analítica"),
            (num(ev["auroc"], 3), f'data-count="{ev["auroc"]}" data-dec="3"', f"AUROC en la prueba reservada de {num(n_test)} pacientes"),
            (num(ev["brier_score"], 3), f'data-count="{ev["brier_score"]}" data-dec="3"', f"Brier del modelo, frente a {num(null['brier_score'], 3)} del modelo nulo"),
            (num(ev["sensibilidad"] * 100) + "%", f'data-count="{round(ev["sensibilidad"] * 100)}" data-suf="%"', f"sensibilidad con el umbral técnico de {num(bundle.get('threshold', .5), 2)}"),
        ]
    else:
        facts = [("—", "", "Genera el modelo para ver sus cifras")]
    rows = "".join(f'<div class="or-fact rv"><b {attrs}>{value}</b><span>{html.escape(label)}</span></div>' for value, attrs, label in facts)
    section(f'''
    <section class="or-split">
      <div class="photo"><img src="{MEDIA}/f23_cientifica.jpg" alt="Investigadora observando una muestra en el microscopio"></div>
      <div class="or-facts"><h2 class="or-d rv">El modelo, en cifras</h2>{rows}</div>
    </section>''')


def render_wide():
    section(f'''
    <section class="or-wide" id="or-wide">
      {video("v7_microscopios", 'id="or-parallax"')}
      <div class="or-wide-in">
        <h2 class="or-d rv">Investigación con trazabilidad</h2>
        <p class="rv">Cada registro conserva su procedencia: desde que entra a Snowflake hasta que el modelo emite una estimación.</p>
      </div>
    </section>
    <div class="or-marquee" aria-hidden="true"><div>{"Ingesta<i>/</i>RAW<i>/</i>CLEAN<i>/</i>ANALYTICAL<i>/</i>Modelado<i>/</i>Evaluación<i>/</i>" * 2}</div></div>''')


def render_method():
    steps = [
        ("01 · Ingesta", "El dato entra con su historia", "El CSV se sube a un Stage de Snowflake y se carga con <code>COPY INTO</code>. Cada carga queda registrada con su archivo, sus filas y su fecha."),
        ("02 · Transformación", "Reglas claras, sin atajos", "SQL construye las capas CLEAN y ANALYTICAL: quita duplicados, valores imposibles y variables posteriores al diagnóstico. La bitácora guarda cada regla."),
        ("03 · Modelado", "Tres modelos, una comparación justa", "Regresión logística, Random Forest y XGBoost con validación cruzada agrupada por paciente. El umbral se elige solo con datos de desarrollo."),
        ("04 · Evaluación", "Una sola prueba, sin repetir", "El modelo elegido se evalúa una vez en los pacientes reservados: discriminación, calibración y clasificación."),
    ]
    articles = "".join(f'<article class="or-step{" on" if i == 0 else ""}" data-i="{i}"><span class="or-label">{lab}</span><h3 class="or-d">{title}</h3><p>{body}</p></article>'
                       for i, (lab, title, body) in enumerate(steps))
    section(f'''
    <section class="or-method" id="metodo">
      <div class="media" aria-hidden="true">
        {video("v2_tubo", 'class="on"')}
        <img src="{MEDIA}/f27_microscopio.jpg" alt="">
        {video("v17_medica")}
        <img src="{MEDIA}/f23_cientifica.jpg" alt="">
        <span class="idx" id="or-idx">01</span>
      </div>
      <div class="or-steps">{articles}</div>
    </section>''')


def render_close():
    section(f'''
    <section class="or-close">
      {video("v17_medica")}
      <div class="or-close-in">
        <h2 class="or-d rv">Los datos también cuentan</h2>
        <p class="rv">Ocho vistas de análisis estadístico: demografía, distribuciones, correlaciones, contrastes, evolución por semestre, PCA y evaluación del modelo.</p>
        <a class="or-link rv" href="?vista=estadisticas" target="_self">Abrir el análisis estadístico <span>→</span></a>
      </div>
    </section>''')


def render_footer():
    home = "?" if st.query_params.get("vista") == "estadisticas" else ""
    section(f'''
    <footer class="or-footer">
      <div class="logo">OncoRisk</div>
      <div>Prototipo académico de ingeniería de datos e inteligencia artificial. Datos sintéticos, validación interna, uso experimental. No emite diagnósticos.</div>
      <div>Explorar<a href="{home}#proyecto" target="_self">Proyecto</a><a href="{home}#metodo" target="_self">Método</a><a href="{home}#explorar" target="_self">Explorar un caso</a><a href="?vista=estadisticas" target="_self">Análisis estadístico</a></div>
      <div>Universidad del Norte<br>Barranquilla, Colombia<br><br>Videos y fotos: Pexels</div>
    </footer>''')


def render_motion_script():
    """Efectos de desplazamiento: aparición, cifras, manifiesto, método fijo y parallax."""
    # Nota: DOMPurify descarta un <script> si su texto contiene etiquetas HTML;
    # por eso el script arma los elementos con la API del DOM y no con cadenas HTML.
    st.html('''<div class="or-motion" hidden></div>
<script>
(() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  document.querySelectorAll('.or video').forEach(v => { v.muted = true; const p = v.play(); if (p) p.catch(() => {}); });

  const man = $('#or-manifiesto');
  if (man && !man.dataset.split) {
    man.dataset.split = '1';
    const frag = document.createDocumentFragment();
    const addWords = (text, hl) => text.split(/(\\s+)/).forEach(part => {
      if (!part) return;
      if (/^\\s+$/.test(part)) { frag.appendChild(document.createTextNode(' ')); return; }
      const s = document.createElement('span'); s.className = hl ? 'w hl' : 'w'; s.textContent = part; frag.appendChild(s);
    });
    [...man.childNodes].forEach(n => addWords(n.textContent, n.nodeType === 1 && n.classList.contains('hl')));
    man.replaceChildren(frag);
  }

  const fmt = (v, dec, sep) => { let s = dec ? v.toFixed(dec).replace('.', ',') : Math.round(v).toString(); return sep ? s.replace(/\\B(?=(\\d{3})+(?!\\d))/g, '.') : s; };
  const count = el => { if (el.dataset.done) return; el.dataset.done = 1; const end = +el.dataset.count, dec = +el.dataset.dec || 0, sep = el.dataset.sep, suf = el.dataset.suf || '';
    if (reduce) { el.textContent = fmt(end, dec, sep) + suf; return; } const t0 = performance.now();
    (function f(t) { const k = Math.min(1, (t - t0) / 1600), e = 1 - Math.pow(1 - k, 4); el.textContent = fmt(end * e, dec, sep) + suf; if (k < 1) requestAnimationFrame(f); })(t0); };
  const io = new IntersectionObserver(es => es.forEach(e => { if (!e.isIntersecting) return; const n = e.target; n.classList.add('in');
    n.querySelectorAll('[data-count]').forEach(count); io.unobserve(n); }), { threshold: .25 });
  $$('.or .rv, .or-split .photo').forEach(el => io.observe(el));

  const media = $$('.or-method .media > video, .or-method .media > img'), idx = $('#or-idx');
  const stepIo = new IntersectionObserver(es => es.forEach(e => { if (!e.isIntersecting) return; const i = +e.target.dataset.i;
    $$('.or-step').forEach(s => s.classList.toggle('on', s === e.target)); media.forEach((m, j) => m.classList.toggle('on', j === i));
    if (idx) idx.textContent = String(i + 1).padStart(2, '0'); }), { rootMargin: '-45% 0px -45% 0px' });
  $$('.or-step').forEach(s => stepIo.observe(s));

  const head = $('#or-header'), hero = $('#or-hero-video'), par = $('#or-parallax'), wide = $('#or-wide');
  const dark = $$('.or-hero, .or-wide, .or-close, .or-stats-hero');
  const words = man ? $$('.w', man) : [];
  function tick() {
    const h = innerHeight;
    if (hero && !reduce) { const r = hero.parentElement.getBoundingClientRect(), y = Math.max(0, -r.top); hero.style.transform = `scale(${1.08 + y / h * .12}) translateY(${y * .25}px)`; }
    if (par && wide && !reduce) { const r = wide.getBoundingClientRect(); par.style.transform = `translateY(${(r.top / h) * -9}%)`; }
    if (man) { const r = man.getBoundingClientRect(), p = Math.min(1, Math.max(0, (h * .85 - r.top) / (r.height + h * .35))), lit = Math.round(p * words.length);
      words.forEach((w, i) => w.classList.toggle('on', i < lit)); }
    if (head && !head.dataset.fixedSolid) head.classList.toggle('solid', !dark.some(s => { const r = s.getBoundingClientRect(); return r.top <= 40 && r.bottom > 40; }));
  }
  if (head && head.classList.contains('solid') && !dark.length) head.dataset.fixedSolid = 1;
  if (!window.__orTick) { window.__orTick = true; document.addEventListener('scroll', () => requestAnimationFrame(() => window.__orRun && window.__orRun()), { capture: true, passive: true }); }
  window.__orRun = tick; tick();
})();
</script>''', unsafe_allow_javascript=True)


# --------------------------------------------------------------------------- herramientas

def patient_input_form() -> pd.DataFrame | None:
    with st.form("formulario_paciente", border=False):
        col1, col2, col3 = st.columns(3, gap="medium")
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
        submitted = st.form_submit_button("Calcular estimación →", width="stretch")
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
    proba = float(bundle["pipeline"].predict_proba(patient_df[bundle["feature_columns"]])[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    model_name = html.escape(MODEL_NAMES.get(bundle["model_name"], bundle["model_name"]))
    st.metric("Probabilidad estimada", f"{proba:.1%}")
    st.html(f'''<div class="or">
      <div class="or-track"><i style="width:{min(max(proba, 0), 1) * 100:.1f}%"></i><s style="left:{threshold * 100:.0f}%"></s></div>
      <div class="or-scale"><span>0 %</span><span>umbral {threshold * 100:.0f} %</span><span>100 %</span></div>
      <dl class="or-dl"><dt>Modelo</dt><dd>{model_name}</dd><dt>Umbral</dt><dd>{num(threshold, 2)} (elegido en desarrollo)</dd>
        <dt>Clasificación</dt><dd>{"Igual o por encima del umbral" if proba >= threshold else "Por debajo del umbral"}</dd></dl>
      <p class="or-note">Probabilidad de clasificación de una condición simulada. No es un diagnóstico ni un riesgo clínico validado.</p></div>''')
    st.caption("Corresponde a los valores enviados en el último cálculo.")
    with st.expander("Valores utilizados en el cálculo"):
        st.html(record_table(patient_df.iloc[0], VARIABLE_LABELS))


def render_explore(bundle: dict | None):
    with st.container(key="explorar"):
        st.html('<div class="or" id="explorar"></div>')
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            st.html('<div class="or or-explore-head"><h2 class="or-d">Explora un caso</h2>'
                    '<p>Ingresa las variables de un paciente sintético y observa la probabilidad que asigna el modelo. Todos los campos son obligatorios.</p></div>')
            if bundle is None:
                st.info("Para habilitar las estimaciones, genera el modelo desde la carpeta del proyecto.", icon=":material/info:")
                st.code("python scripts/run_pipeline.py --backend local --usar-csv-local", language="bash")
            else:
                patient_df = patient_input_form()
                if patient_df is not None:
                    st.session_state["oncorisk_input"] = patient_df
        with right, st.container(key="resultado"):
            if bundle is not None and "oncorisk_input" in st.session_state:
                render_prediction(bundle, st.session_state["oncorisk_input"])
            else:
                st.html('<div class="or or-out-empty"><span class="or-label">Probabilidad estimada</span><div class="big or-d">—</div>'
                        '<p>Completa las variables y selecciona «Calcular estimación». El resultado aparecerá aquí.</p></div>')


def readable_feature(name: str) -> str:
    """Traduce también las columnas codificadas que expone el modelo."""
    feature = name.split("__", 1)[-1]
    for original in sorted(VARIABLE_LABELS, key=len, reverse=True):
        if feature == original:
            return VARIABLE_LABELS[original]
        if feature.startswith(original + "_"):
            value = feature[len(original) + 1:]
            return f"{VARIABLE_LABELS[original]}: {category(value, original)}"
    return feature.replace("_", " ")


def render_interpretation(bundle: dict | None):
    with st.container(key="modelo"):
        name = html.escape(MODEL_NAMES.get(bundle["model_name"], bundle["model_name"])) if bundle else "Modelo pendiente"
        st.html(f'<div class="or or-model-head" id="modelo"><h2 class="or-d">{name}</h2>'
                '<p>Seleccionado entre regresión logística, Random Forest y XGBoost mediante validación cruzada agrupada por paciente.</p></div>')
        if bundle is None:
            st.caption("La interpretación y las métricas estarán disponibles después de generar el modelo.")
            return
        with st.expander("Variables con mayor contribución global"):
            st.write("Describen el comportamiento general del modelo. No explican por sí solas un resultado individual ni indican causalidad.")
            explanation = explain_model(bundle["pipeline"], top_n=10)
            explanation["variable"] = explanation["variable"].map(readable_feature)
            value_column = explanation.columns[1]
            values = [round(float(v), 3) for v in explanation[value_column]][::-1]
            st.echarts_chart({"grid": {"left": 4, "right": 44, "top": 6, "bottom": 4, "containLabel": True}, "tooltip": {"trigger": "item"},
                              "xAxis": {"type": "value", **AXIS},
                              "yAxis": {"type": "category", "data": explanation["variable"].tolist()[::-1], **AXIS, "axisLabel": {"color": "#16231d", "fontSize": 12}, "splitLine": {"show": False}},
                              "series": [{"type": "bar", "barWidth": "60%", "label": {"show": True, "position": "right", "formatter": "{c}", "fontSize": 11},
                                          "data": [{"value": v, "itemStyle": {"color": "#ff8a5c" if v > 0 else "#1fb985",
                                                                              "borderRadius": [0, 6, 6, 0] if v > 0 else [6, 0, 0, 6]}} for v in values]}]},
                             height=320, alt="Contribución de las variables al modelo")
            st.html('<p class="dk-note">Naranja aumenta la probabilidad estimada; verde la disminuye.</p>')
            rows = [[html.escape(r.variable), f"<b>{num(float(getattr(r, '_2')), 3)}</b>", "↑ aumenta" if float(getattr(r, '_2')) > 0 else "↓ disminuye"]
                    for r in explanation.itertuples()]
            st.html('<div class="dv-card">' + _table(["Variable", "Coeficiente (log-momios)", "Efecto en la estimación"], rows, {1}) + "</div>")
        with st.expander("Desempeño en la prueba reservada"):
            ev, ref = bundle["evaluation"]["modelo"], bundle["evaluation"]["modelo_nulo"]
            pct = lambda v: f"{float(v) * 100:.1f}".replace(".", ",") + " %"
            rows = [["AUROC", f"<b>{num(ev['auroc'], 3)}</b>", "0,500"], ["PR-AUC (precisión promedio)", f"<b>{num(ev['pr_auc'], 3)}</b>", num(ref['pr_auc'], 3)],
                    ["Brier", f"<b>{num(ev['brier_score'], 3)}</b>", num(ref['brier_score'], 3)], ["Sensibilidad", f"<b>{pct(ev['sensibilidad'])}</b>", "—"],
                    ["Especificidad", f"<b>{pct(ev['especificidad'])}</b>", "—"], ["F1", f"<b>{num(ev['f1'], 3)}</b>", "—"],
                    ["Umbral técnico", f"<b>{num(bundle.get('threshold', 0.5), 2)}</b>", "—"]]
            st.html('<div class="dv-card">' + _table(["Métrica", "Modelo", "Modelo nulo"], rows, {1, 2}) + "</div>")
            st.caption("Evaluación interna con datos sintéticos. Un Brier menor indica mejor desempeño probabilístico; no demuestra utilidad clínica. "
                       "El umbral se eligió maximizando F1 con predicciones fuera de pliegue del conjunto de desarrollo; no es un punto de corte clínico.")


ORIGINS = {"snowflake_tabla_prueba": "Datos de Snowflake", "csv_sintetico_local": "CSV local", "csv_cargado": "CSV subido", "generador_sintetico_pruebas": "Lote generado"}


def origin_label(value) -> str:
    key = str(value or "").split(":")[0]
    return ORIGINS.get(key, key.replace("_", " ") or "Sin registro")


def render_model_status(bundle: dict | None):
    """Estado del modelo en producción e historial de entrenamientos (registro de experimentos)."""
    from datetime import datetime
    from src.modeling.registry import read_experiments

    if bundle is not None:
        ev, null = bundle["evaluation"]["modelo"], bundle["evaluation"]["modelo_nulo"]
        trained = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
        origin = origin_label(bundle.get("data_origin"))
        cells = [("Modelo", html.escape(MODEL_NAMES.get(bundle["model_name"], bundle["model_name"])), f"entrenado {trained}"),
                 ("AUROC", num(ev["auroc"], 3), "prueba reservada"),
                 ("Brier", num(ev["brier_score"], 3), f"nulo {num(null['brier_score'], 3)}"),
                 ("Umbral", num(bundle.get("threshold", .5), 2), html.escape(origin))]
        st.html('<div class="or-kpis">' + "".join(f'<div class="or-kpi"><span>{k}</span><b>{v}</b><small>{s}</small></div>' for k, v, s in cells) + "</div>")
    runs = [r for r in read_experiments() if r.get("test_metrics")]
    if runs:
        history = pd.DataFrame([{"entrenamiento": i + 1, "fecha": r["timestamp_utc"][:16].replace("T", " "),
                                 "Brier": r["test_metrics"].get("brier_score"), "AUROC": r["test_metrics"].get("auroc"),
                                 "modelo": MODEL_NAMES.get(r["model_name"], r["model_name"]),
                                 "origen": origin_label(r.get("origen_datos"))} for i, r in enumerate(runs)])
        null_brier = float(bundle["evaluation"]["modelo_nulo"]["brier_score"]) if bundle else None
        st.html('<div class="or-dash-h"><h3>Historial de entrenamientos</h3><p>Brier del modelo seleccionado en cada entrenamiento. La línea punteada es el modelo nulo: por debajo de ella, el modelo calibra mejor que asignar la prevalencia a todos.</p></div>')
        layers = [{"mark": {"type": "line", "color": "#2048ff", "strokeWidth": 2.5, "point": {"filled": True, "size": 70, "color": "#2048ff"}},
                   "encoding": {"x": {"field": "entrenamiento", "type": "ordinal", "title": "Entrenamiento", "axis": {"labelAngle": 0}},
                                "y": {"field": "Brier", "type": "quantitative", "scale": {"zero": False}, "title": "Brier"},
                                "tooltip": [{"field": "fecha", "title": "Fecha (UTC)"}, {"field": "modelo"}, {"field": "Brier", "type": "quantitative", "format": ".3f"},
                                            {"field": "AUROC", "type": "quantitative", "format": ".3f"}, {"field": "origen", "title": "Origen de datos"}]}}]
        if null_brier is not None:
            layers.append({"mark": {"type": "rule", "color": "#ff5a1f", "strokeDash": [5, 4], "strokeWidth": 1.5}, "data": {"values": [{"y": null_brier}]},
                           "encoding": {"y": {"field": "y", "type": "quantitative"}}})
        series = [{"type": "line", "encode": {"x": "entrenamiento", "y": "Brier"}, "symbolSize": 9, "lineStyle": {"width": 3, "color": "#14946a"},
                   "itemStyle": {"color": "#14946a", "borderColor": "#fff", "borderWidth": 2}}]
        if null_brier is not None:
            series[0]["markLine"] = {"silent": True, "symbol": "none", "lineStyle": {"color": "#ff8a5c", "type": "dashed"},
                                     "data": [{"yAxis": round(null_brier, 4)}], "label": {"formatter": "modelo nulo", "color": "#7b8a83"}}
        frame = history.assign(entrenamiento=history.entrenamiento.astype(str), Brier=history.Brier.astype(float).round(4))
        st.echarts_chart({"grid": {"left": 4, "right": 70, "top": 16, "bottom": 4, "containLabel": True}, "tooltip": {"trigger": "axis"},
                          "dataset": {"source": frame[["entrenamiento", "Brier"]]}, "xAxis": {"type": "category", **AXIS, "splitLine": {"show": False}},
                          "yAxis": {"type": "value", "scale": True, **AXIS}, "series": series}, height=230, alt="Brier del modelo en cada entrenamiento")
        with st.expander("Ver los entrenamientos registrados"):
            rows = [[str(r.entrenamiento), html.escape(r.fecha), html.escape(r.modelo), f"<b>{num(r.Brier, 3)}</b>", num(r.AUROC, 3), html.escape(r.origen)]
                    for r in history.iloc[::-1].itertuples()]
            st.html('<div class="dv-card">' + _table(["#", "Fecha (UTC)", "Modelo", "Brier", "AUROC", "Origen"], rows, {0, 3, 4}) + "</div>")


ADMIN_MAX_INTENTOS = 5
ADMIN_BLOQUEO_SEG = 300


def admin_desbloqueado() -> bool:
    """Pide el código de ADMIN_PANEL_CODE. Sin código configurado el panel queda cerrado (nunca abierto)."""
    esperado = os.getenv("ADMIN_PANEL_CODE", "").strip()
    if not esperado:
        st.warning("El panel está bloqueado: no hay código de administrador configurado. "
                   "Define ADMIN_PANEL_CODE en el archivo .env y reinicia la página.")
        return False
    if st.session_state.get("admin_ok"):
        if st.button("Cerrar sesión de administrador", key="admin_logout"):
            st.session_state["admin_ok"] = False
            st.rerun()
        return True
    hasta = st.session_state.get("admin_bloqueado_hasta", 0.0)
    if time.time() < hasta:
        st.error(f"Demasiados intentos. Espera {int(hasta - time.time()) + 1} s para volver a intentar.")
        return False
    ingresado = st.text_input("Código de administrador", type="password", key="admin_code_input")
    if not ingresado:
        st.caption("Ingresa el código de administrador para desbloquear esta sección.")
        return False
    if hmac.compare_digest(ingresado.encode(), esperado.encode()):
        st.session_state["admin_ok"] = True
        st.session_state["admin_fallos"] = 0
        st.rerun()
    fallos = st.session_state.get("admin_fallos", 0) + 1
    st.session_state["admin_fallos"] = fallos
    if fallos >= ADMIN_MAX_INTENTOS:
        st.session_state["admin_bloqueado_hasta"] = time.time() + ADMIN_BLOQUEO_SEG
        st.session_state["admin_fallos"] = 0
        st.error("Demasiados intentos. El acceso se bloqueó por 5 minutos.")
    else:
        st.error("Código incorrecto.")
    return False


def render_admin_panel(bundle: dict | None = None):
    with st.container(key="admin"), st.expander("Acceso administración", icon=":material/lock:"):
        st.html('<div class="or-dash-h"><h3>Panel de administración</h3><p>Actualiza el modelo con datos nuevos y revisa su desempeño.</p></div>')
        if not admin_desbloqueado():
            return

        if st.session_state.get("admin_resultado"):
            st.success(st.session_state["admin_resultado"])
        render_model_status(bundle)
        st.html('<div class="or-dash-h" style="margin-top:22px"><h3>Reentrenar el modelo</h3><p>Ejecuta el pipeline completo sin usar la terminal. El modelo nuevo reemplaza al actual y queda en el historial.</p></div>')
        fuente = st.radio("Fuente de datos", ["Snowflake", "Generar lote local pequeño"], index=0, key="admin_fuente")
        password = ""
        uses_token = os.getenv("SNOWFLAKE_AUTHENTICATOR", "PROGRAMMATIC_ACCESS_TOKEN").strip().upper() == "PROGRAMMATIC_ACCESS_TOKEN"
        secret_label = "Token PAT de Snowflake" if uses_token else "Contraseña de Snowflake"
        if fuente.startswith("Snowflake"):
            password = st.text_input(secret_label, type="password", key="admin_password",
                                     help="No se guarda; solo se usa para esta actualización.")
        archivo, problemas = None, []
        if True:
            archivo = st.file_uploader("Subir o cambiar CSV (opcional)", type="csv", key="admin_csv",
                                       help="En Snowflake reemplaza la tabla de entrada al entrenar; sin archivo se usa la tabla actual. En local, sin archivo se genera un lote pequeño.")
            st.caption("El CSV debe tener las mismas columnas del lote sintético: id_paciente, edad, sexo, "
                       "antecedente_familiar_cancer, antecedente_personal_benigno, tabaquismo, imc, presion_sistolica, "
                       "glucosa_mg_dl, paridad, ca125_u_ml, resultado_imagen, fuente, periodo y riesgo_oncologico (0/1). "
                       "Los identificadores directos (nombre, documento) no se guardan.")
            if archivo is not None:
                try:
                    cargado = read_uploaded_csv(io.BytesIO(archivo.getvalue()))
                    problemas = validate_uploaded_frame(cargado)
                except Exception as exc:
                    problemas = [f"No se pudo leer el archivo: {exc}"]
                if problemas:
                    st.error(" ".join(problemas))
                else:
                    st.success(f"{archivo.name}: {len(cargado):,} registros listos para entrenar.")
        confirmar = True
        if fuente.startswith("Snowflake") and archivo is not None:
            confirmar = st.checkbox("Confirmo reemplazar los datos de entrada de Snowflake con este CSV", key="admin_confirmar_csv")
        if st.button("Entrenar / actualizar modelo", width="stretch", key="admin_entrenar"):
            if fuente.startswith("Snowflake") and not password:
                st.error(f"Ingresa el {secret_label[0].lower() + secret_label[1:]}.")
                return
            if not confirmar:
                st.error("Confirma el reemplazo de la tabla de Snowflake.")
                return
            if problemas:
                st.error("Corrige el archivo CSV antes de entrenar.")
                return
            if password:
                # Con autenticación PAT el conector lee SNOWFLAKE_TOKEN, no SNOWFLAKE_PASSWORD.
                os.environ["SNOWFLAKE_TOKEN" if uses_token else "SNOWFLAKE_PASSWORD"] = password
            with st.spinner("Entrenando modelo... puede tardar uno o dos minutos."):
                tmp_path = None
                try:
                    usa_snowflake = fuente.startswith("Snowflake")
                    extra = {}
                    if archivo is not None:
                        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                            tmp.write(archivo.getvalue())
                            tmp_path = tmp.name
                        if usa_snowflake:
                            from src.data_layer.upload_snowflake import replace_uploaded_csv
                            replace_uploaded_csv(tmp_path)
                            st.session_state["admin_resultado"] = "CSV actualizado en Snowflake. Entrenamiento pendiente de finalizar."
                        extra = {"uploaded_csv": tmp_path, "uploaded_name": archivo.name}
                    summary = run_full_pipeline(backend="snowflake" if usa_snowflake else "local", use_snowflake_csv=usa_snowflake, **extra)
                except Exception as exc:
                    st.error(f"No se pudo entrenar: {exc}")
                    return
                finally:
                    if tmp_path:
                        os.unlink(tmp_path)
            load_model_bundle.clear()
            ev = summary["evaluacion_prueba_reservada"]["modelo"]
            st.session_state["admin_resultado"] = f"Modelo actualizado: {summary['modelo_seleccionado']} (AUROC {ev['auroc']:.3f})"
            st.rerun()


def main():
    st.html(APP_DIR / "portal.css")
    bundle = load_model_bundle()
    if st.query_params.get("vista") == "estadisticas":
        render_header(solid=True, stats=True)
        render_statistics_panel(bundle)
        render_footer()
        render_motion_script()
        return
    render_header(solid=False)
    render_hero()
    render_statement()
    render_facts(bundle)
    render_wide()
    render_method()
    render_explore(bundle)
    render_interpretation(bundle)
    render_close()
    render_footer()
    render_admin_panel(bundle)
    render_motion_script()


if __name__ == "__main__":
    main()