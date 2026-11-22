"""Las cuatro vistas del análisis detallado: cohorte, factores asociados, modelo y calidad.

Cada vista abre con la idea que debe transmitir y usa st.echarts_chart con el mismo
estilo del resumen. Las tablas se dibujan como HTML liviano para que combinen con
las tarjetas del panel.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import streamlit as st

from app.dashboard_cards import AXIS, CORAL, GREEN, GREEN2, INK, LIME, MUTE, _miles, _pct
from config import ANALYTICAL_DIR, CLEAN_DIR, RAW_DIR

VIEWS = ["Cohorte", "Distribuciones", "Factores asociados", "Modelo", "Calidad de datos"]


def _num(v, d: int = 1) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _msg(text: str) -> None:
    st.html(f'<p class="dv-msg">{text}</p>')


def _table(head: list[str], rows: list[list[str]], numeric: set[int] = frozenset()) -> str:
    th = "".join(f'<th class="{"n" if i in numeric else ""}">{h}</th>' for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<table class="dv-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def record_table(row: pd.Series, labels: dict | None = None) -> str:
    """Una fila de datos mostrada como lista Variable / Valor."""
    labels = labels or {}
    rows = [[html.escape(str(labels.get(k, k))), f"<b>{html.escape(_num(v, 2) if isinstance(v, float) else str(v))}</b>"] for k, v in row.items()]
    return '<div class="dv-card">' + _table(["Variable", "Valor"], rows, {1}) + "</div>"


def _card_head(title: str, sub: str) -> None:
    st.html(f'<div class="dk-h"><h3>{title}</h3><p>{sub}</p></div>')


# ------------------------------------------------------------------ 1. Cohorte
def view_cohort(data: pd.DataFrame) -> None:
    n = len(data)
    women = (data.sexo == "F").mean() if n else 0
    q1, q3 = data.edad.quantile([.25, .75]) if n else (0, 0)
    top = data.fuente.value_counts(normalize=True)
    top_name = {"consulta_externa": "consulta externa", "remision": "remisión", "urgencias": "urgencias"}.get(top.index[0], top.index[0]) if len(top) else "—"
    _msg(f"<b>La selección tiene {_miles(n)} pacientes sintéticos</b>; el {_pct(women, 0)} son mujeres, la mitad tiene entre "
         f"{q1:.0f} y {q3:.0f} años y la mayoría llega por {top_name} ({_pct(top.iloc[0] if len(top) else 0, 0)}).")
    st.caption(f"{_miles(n)} pacientes en la selección")
    a, b, c = st.columns([1.2, 1, 1.05], gap="medium")
    with a, st.container(key="dv_age"):
        _card_head("Edad por sexo", "Pacientes en cada rango de edad")
        bins = [18, 30, 40, 50, 60, 70, 80, 200]
        names = ["18–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80+"]
        rango = pd.cut(data.edad, bins, right=False, labels=names)
        table = (pd.crosstab(rango, data.sexo).reindex(names, fill_value=0)
                 .rename(columns={"F": "Mujeres", "M": "Hombres"}).reset_index().rename(columns={"edad": "rango"}))
        table.columns = [str(col) for col in table.columns]
        series = [{"type": "bar", "itemStyle": {"color": color, "borderRadius": [6, 6, 0, 0]}, "barGap": "15%"}
                  for col, color in (("Mujeres", GREEN2), ("Hombres", LIME)) if col in table]
        st.echarts_chart({"grid": {"left": 4, "right": 4, "top": 30, "bottom": 4, "containLabel": True},
                          "legend": {"top": 0, "right": 0, "itemWidth": 10, "itemHeight": 10},
                          "tooltip": {"trigger": "axis"}, "dataset": {"source": table},
                          "xAxis": {"type": "category", **AXIS, "splitLine": {"show": False}}, "yAxis": {"type": "value", **AXIS},
                          "series": series}, height=260, alt="Pacientes por rango de edad y sexo")
    with b, st.container(key="dv_source"):
        _card_head("Procedencia", "Fuente del registro")
        src = data.fuente.map({"consulta_externa": "Consulta externa", "remision": "Remisión", "urgencias": "Urgencias"}).value_counts()
        st.echarts_chart({"tooltip": {"trigger": "item"}, "color": [GREEN, GREEN2, LIME],
                          "legend": {"bottom": 0, "itemWidth": 10, "itemHeight": 10},
                          "series": [{"type": "pie", "radius": ["50%", "74%"], "center": ["50%", "44%"], "padAngle": 2,
                                      "itemStyle": {"borderRadius": 6}, "label": {"show": False},
                                      "data": [{"name": k, "value": int(v)} for k, v in src.items()]}]},
                         height=260, alt="Distribución de pacientes por procedencia")
    with c, st.container(key="dv_vars"):
        _card_head("Variables clínicas", "Resumen de la selección")
        rows = []
        for col, name, d in (("edad", "Edad (años)", 1), ("imc", "IMC (kg/m²)", 1), ("glucosa_mg_dl", "Glucosa (mg/dL)", 0),
                             ("presion_sistolica", "Presión sistólica (mmHg)", 0), ("ca125_u_ml", "CA-125 (U/mL)", 1)):
            s = pd.to_numeric(data[col], errors="coerce").dropna()
            rows.append([name, f"<b>{_num(s.mean(), d)}</b>", _num(s.median(), d), f"{_num(s.min(), 0)} – {_num(s.max(), 0)}"])
        st.html(_table(["Variable", "Media", "Mediana", "Rango"], rows, {1, 2, 3}))
    class_balance(data)


# ------------------------------------------------------- 2. Factores asociados
FACTOR_DEFS = [
    ("Imagen sospechosa", lambda d: d.resultado_imagen == "sospechoso"),
    ("Antecedente familiar", lambda d: d.antecedente_familiar_cancer == 1),
    ("Tabaquismo activo", lambda d: d.tabaquismo == "activo"),
    ("Edad 70+", lambda d: d.edad >= 70),
    ("CA-125 > 35", lambda d: d.ca125_u_ml > 35),
    ("Antecedente benigno", lambda d: d.antecedente_personal_benigno == 1),
    ("Obesidad (IMC ≥ 30)", lambda d: d.imc >= 30),
    ("Imagen normal", lambda d: d.resultado_imagen == "normal"),
]


def odds_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """OR crudas con IC log-Wald del 95 %; se suma 0,5 a la tabla si hay celdas nulas."""
    d = data.loc[data.riesgo_oncologico.isin([0, 1])]
    y = d.riesgo_oncologico == 1
    rows = []
    for name, rule in FACTOR_DEFS:
        m = rule(d).fillna(False)
        a, b, c, e = [float(v) for v in ((m & y).sum(), (m & ~y).sum(), (~m & y).sum(), (~m & ~y).sum())]
        if min(a, b, c, e) == 0:
            a, b, c, e = a + .5, b + .5, c + .5, e + .5
        o = a * e / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / e)
        rows.append({"factor": name, "or": o, "ic_inf": float(np.exp(np.log(o) - 1.96 * se)), "ic_sup": float(np.exp(np.log(o) + 1.96 * se)),
                     "con_factor": int(m.sum())})
    return pd.DataFrame(rows).sort_values("or", ascending=False).reset_index(drop=True)


def view_factors(data: pd.DataFrame) -> None:
    labels = set(data.riesgo_oncologico.dropna().unique())
    if not {0, 1} <= labels:
        st.info("Las razones de probabilidades requieren pacientes con etiqueta 0 y 1 en la selección. Agrega ambas etiquetas en el filtro.")
        return
    ors = odds_ratios(data)
    top = ors.iloc[0]
    risk = ors.loc[ors["or"] > 1, "factor"].tolist()
    _msg(f"<b>{top.factor.capitalize()} multiplica por {_num(top['or'], 1)} la probabilidad de etiqueta 1</b>"
         f"{'; le siguen ' + ', '.join(f.lower() for f in risk[1:4]) if len(risk) > 1 else ''}. Esto responde la pregunta problema.")
    left, right = st.columns([1.4, 1], gap="medium")
    with left, st.container(key="dv_forest"):
        _card_head("Razón de probabilidades (OR) con IC del 95 %", "A la derecha de la línea punteada, el factor se asocia con más riesgo; a la izquierda, con menos")
        plot = ors.iloc[::-1].reset_index(drop=True)
        lines = [{"type": "line", "silent": True, "symbol": "none", "data": [[float(r.ic_inf), i], [float(r.ic_sup), i]],
                  "lineStyle": {"width": 3, "color": "#f6b79c" if r["or"] > 1 else "#9fdcc4"}} for i, r in plot.iterrows()]
        points = {"type": "scatter", "symbolSize": 14,
                  "data": [{"value": [round(float(r["or"]), 2), i], "itemStyle": {"color": CORAL if r["or"] > 1 else GREEN}} for i, r in plot.iterrows()],
                  "label": {"show": True, "position": "right", "formatter": "{@[0]}", "color": INK, "fontWeight": 700},
                  "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": INK, "type": "dashed"}, "data": [{"xAxis": 1}],
                               "label": {"formatter": "1", "color": MUTE}}}
        st.echarts_chart({"grid": {"left": 4, "right": 36, "top": 10, "bottom": 4, "containLabel": True},
                          "tooltip": {"trigger": "item"},
                          "xAxis": {"type": "log", "min": .2, "max": max(6.0, float(ors.ic_sup.max()) * 1.1), **AXIS},
                          "yAxis": {"type": "category", "data": plot.factor.tolist(), **AXIS, "axisLabel": {"color": INK, "fontSize": 12}, "splitLine": {"show": False}},
                          "series": lines + [points]}, height=340, alt="Razón de probabilidades de cada factor con su intervalo del 95 %")
    with right, st.container(key="dv_factors_table"):
        _card_head("Lectura", "Comparado con quienes no tienen el factor")
        rows = [[html.escape(r.factor), f"<b>{_num(r['or'], 2)}</b>", f"{_num(r.ic_inf, 2)} – {_num(r.ic_sup, 2)}",
                 '<span class="dv-pill up">↑ riesgo</span>' if r.ic_inf > 1 else ('<span class="dv-pill dn">↓ riesgo</span>' if r.ic_sup < 1 else '<span class="dv-pill">sin diferencia</span>')]
                for _, r in ors.iterrows()]
        st.html(_table(["Factor", "OR", "IC 95 %", ""], rows, {1, 2}) + '<p class="dk-note">OR crudas con IC log-Wald; asociación en datos sintéticos, no causalidad.</p>')


# ------------------------------------------------------------------- 3. Modelo
def view_model(bundle, diagnostics, data_full: pd.DataFrame) -> None:
    if bundle is None:
        st.warning("No hay un modelo guardado disponible.")
        return
    threshold = float(bundle.get("threshold", .5))
    try:
        with st.spinner("Comprobando la prueba reservada y sus intervalos…"):
            results, holdout = diagnostics(bundle, threshold)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"No se puede verificar la evaluación: {exc}")
        return
    m = results["metrics"].set_index("Métrica")
    null_brier = float(bundle["evaluation"]["modelo_nulo"]["brier_score"])
    auc, brier = float(m.loc["ROC-AUC", "Estimación"]), float(m.loc["Brier", "Estimación"])
    ok = auc > .5 and brier < null_brier
    _msg(f"<b>{'Se cumple la hipótesis' if ok else 'La hipótesis no se cumple'}</b>: AUROC {_num(auc, 3)} (el azar es 0,50) y Brier "
         f"{_num(brier, 3)} frente a {_num(null_brier, 3)} del modelo nulo, en {_miles(results['n'])} pacientes que el modelo nunca vio.")
    summary = _summary()
    model_comparison(summary)
    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        model_contribution(summary)
    with right:
        model_subgroups(holdout, data_full)
    mat = results["matrix"].set_index(["Etiqueta observada", "Clasificación del modelo"]).Pacientes
    tn, fp, fn, tp = (int(mat[("Etiqueta 0", "Etiqueta 0")]), int(mat[("Etiqueta 0", "Etiqueta 1")]),
                      int(mat[("Etiqueta 1", "Etiqueta 0")]), int(mat[("Etiqueta 1", "Etiqueta 1")]))
    a, b = st.columns(2, gap="medium")
    with b:
        pr_curve(results, float(holdout.etiqueta.mean()))
    with a, st.container(key="dv_roc"):
        _card_head("Curva ROC", "Qué tan bien separa los dos grupos")
        roc = results["roc"].rename(columns={"Falsos positivos": "fpr", "Sensibilidad": "tpr"})
        if len(roc) > 400:
            roc = roc.iloc[np.linspace(0, len(roc) - 1, 400).astype(int)]
        point = [round(fp / max(fp + tn, 1), 3), round(tp / max(tp + fn, 1), 3)]
        st.echarts_chart({"grid": {"left": 4, "right": 14, "top": 10, "bottom": 26, "containLabel": True}, "tooltip": {"trigger": "axis"},
                          "dataset": {"source": roc.reset_index(drop=True)},
                          "xAxis": {"type": "value", "min": 0, "max": 1, "name": "1 − especificidad", "nameLocation": "middle", "nameGap": 26, **AXIS},
                          "yAxis": {"type": "value", "min": 0, "max": 1, **AXIS},
                          "series": [{"type": "line", "symbol": "none", "encode": {"x": "fpr", "y": "tpr"}, "lineStyle": {"width": 3, "color": GREEN},
                                      "areaStyle": {"color": "rgba(31,185,133,.15)"}},
                                     {"type": "line", "symbol": "none", "data": [[0, 0], [1, 1]], "lineStyle": {"type": "dashed", "color": "#c5ccd8"}},
                                     {"type": "scatter", "data": [point], "symbolSize": 12, "itemStyle": {"color": CORAL},
                                      "label": {"show": True, "position": "right", "formatter": f"umbral {_num(threshold, 2)}", "color": INK}}]},
                         height=260, alt="Curva ROC del modelo en la prueba reservada")
    b, c = st.columns(2, gap="medium")
    with b, st.container(key="dv_cal"):
        _card_head("Calibración", "Probabilidad estimada frente a la observada")
        cal = results["calibration"].rename(columns={"Probabilidad estimada": "estimada", "Proporción observada": "observada"}).round(4)
        top = float(max(cal.estimada.max(), cal.observada.max(), .1)) * 1.1
        st.echarts_chart({"grid": {"left": 4, "right": 14, "top": 10, "bottom": 26, "containLabel": True}, "tooltip": {"trigger": "item"},
                          "dataset": {"source": cal},
                          "xAxis": {"type": "value", "min": 0, "max": round(top, 2), "name": "Estimada", "nameLocation": "middle", "nameGap": 26, **AXIS},
                          "yAxis": {"type": "value", "min": 0, "max": round(top, 2), **AXIS},
                          "series": [{"type": "line", "symbol": "none", "data": [[0, 0], [round(top, 2), round(top, 2)]], "lineStyle": {"type": "dashed", "color": "#c5ccd8"}},
                                     {"type": "line", "encode": {"x": "estimada", "y": "observada"}, "symbolSize": 9, "lineStyle": {"width": 3, "color": GREEN},
                                      "itemStyle": {"color": GREEN, "borderColor": "#fff", "borderWidth": 2}}]},
                         height=260, alt="Curva de calibración del modelo")
    with c, st.container(key="dv_cm"):
        _card_head("Matriz de confusión", f"Umbral {_num(threshold, 2)}")
        st.echarts_chart({"grid": {"left": 4, "right": 4, "top": 6, "bottom": 4, "containLabel": True},
                          "xAxis": {"type": "category", "data": ["Predice 0", "Predice 1"], **AXIS, "splitLine": {"show": False}},
                          "yAxis": {"type": "category", "data": ["Real 1", "Real 0"], **AXIS, "splitLine": {"show": False}},
                          "visualMap": {"show": False, "min": min(tn, fp, fn, tp), "max": max(tn, fp, fn, tp), "inRange": {"color": ["#e3f4ec", GREEN]}},
                          "series": [{"type": "heatmap", "data": [[0, 1, tn], [1, 1, fp], [0, 0, fn], [1, 0, tp]],
                                      "itemStyle": {"borderColor": "#fff", "borderWidth": 6, "borderRadius": 10},
                                      "label": {"show": True, "fontSize": 20, "fontWeight": 800, "color": INK, "formatter": "{@[2]}"}}]},
                         height=260, alt="Matriz de confusión con verdaderos y falsos positivos y negativos")
    with st.container(key="dv_metrics"):
        _card_head("Métricas en la prueba reservada", "Intervalo del 95 % por bootstrap de 500 réplicas")

        def row(name, label, meaning, pct=False):
            r = m.loc[name]
            f = (lambda v: _pct(v, 1)) if pct else (lambda v: _num(v, 3))
            return [label, f"<b>{f(r['Estimación'])}</b>", f"{f(r['IC 95% inferior'])} – {f(r['IC 95% superior'])}", meaning]
        sens, spec, ppv = float(m.loc["Recall (sensibilidad)", "Estimación"]), float(m.loc["Especificidad", "Estimación"]), float(m.loc["Precision (VPP)", "Estimación"])
        rows = [row("ROC-AUC", "AUROC", "Separa los grupos mejor que el azar (0,50)"),
                row("Brier", "Brier", f"Error de las probabilidades; el modelo nulo tiene {_num(null_brier, 3)}"),
                row("Recall (sensibilidad)", "Sensibilidad", f"De cada 10 pacientes con etiqueta 1, detecta unos {round(sens * 10)}", True),
                row("Especificidad", "Especificidad", f"De cada 10 sin etiqueta 1, descarta unos {round(spec * 10)}", True),
                row("Precision (VPP)", "Precisión (VPP)", f"De los marcados como riesgo, unos {round(ppv * 10)} de cada 10 lo tienen", True),
                row("F1-Score", "F1", "Equilibrio entre precisión y sensibilidad (0 a 1)"),
                row("Precisión promedio (AP)", "PR-AUC", f"Área bajo la curva PR; el azar sería {_num(float(holdout.etiqueta.mean()), 3)}")]
        st.html(_table(["Métrica", "Valor", "IC 95 %", "Qué significa"], rows, {1, 2}))
        st.download_button("Descargar métricas e intervalos", results["metrics"].to_csv(index=False).encode("utf-8-sig"),
                           "evaluacion_prueba_reservada.csv", "text/csv", key="stats_download_model")


# ---------------------------------------------------------- 4. Calidad de datos
def _rows(path) -> int | None:
    try:
        return len(pd.read_parquet(path, columns=None))
    except (OSError, ValueError, ImportError):
        return None


def view_quality(data: pd.DataFrame, report: dict, context: dict, metadata: dict) -> None:
    s = report["summary"]
    complete = 100 * (1 - s["missing_pct"] / 100) if "missing_pct" in s else 100.0
    layers = [("CSV de origen", metadata.get("rows") or s["rows"]), ("RAW", _rows(RAW_DIR / "clinical_raw.parquet")),
              ("CLEAN", _rows(CLEAN_DIR / "clinical_clean.parquet")), ("ANALYTICAL", _rows(ANALYTICAL_DIR / "clinical_analytical.parquet"))]
    counts = [v for _, v in layers if v is not None]
    lost = (max(counts) - min(counts)) if counts else 0
    _msg(f"<b>Cada registro es trazable</b>: {'ninguno se perdió entre capas' if lost == 0 else f'{_miles(lost)} registros se filtraron entre capas'}, "
         f"no hay celdas ausentes{'' if s['missing_cells'] == 0 else ' salvo ' + _miles(s['missing_cells'])} y cada regla quedó en la bitácora de auditoría.")
    kp = [("Registros en la selección", _miles(s["rows"])), ("Completitud de celdas", _num(complete, 1) + " %"),
          ("Identificadores repetidos", _miles(s["duplicate_ids"])), ("Valores fuera de dominio", _miles(s["invalid_cells"]))]
    st.html('<div class="dv-kp">' + "".join(f'<div><small>{k}</small><b>{v}</b></div>' for k, v in kp) + "</div>")
    flow = '<i>→</i>'.join(f'<div><small>{n}</small><b>{_miles(v) if v is not None else "—"}</b></div>' for n, v in layers)
    with st.container(key="dv_flow"):
        _card_head("Recorrido de los datos", "Filas en cada capa del proceso (las capas se replican en Snowflake)")
        st.html(f'<div class="dv-flow">{flow}</div>')
    left, right = st.columns([1, 1.2], gap="medium")
    with left, st.container(key="dv_indicators"):
        _card_head("Indicadores del proceso", "Del archivo recibido al conjunto que usa el modelo")
        first = layers[0][1] or s["rows"]
        final = layers[-1][1] or s["rows"]
        rows = [["Registros iniciales", _miles(first)], ["Pacientes únicos", _miles(s["patients"])],
                ["Registros duplicados", _miles(s["duplicate_rows"])], ["Registros excluidos", _miles(max(first - final, 0))],
                ["Variables", _miles(data.shape[1])], ["Conjunto final (ANALYTICAL)", f"<b>{_miles(final)}</b>"]]
        st.html(_table(["Indicador", "Resultado"], rows, {1}))
    with right:
        missing_chart(report)
    outliers_chart(report)
    with st.container(key="dv_issues"):
        _card_head("Revisiones de calidad", "Señales para revisar; no se borran ni se imputan registros")
        issues = report["issues"]
        names = {"edad": "Edad", "imc": "IMC", "presion_sistolica": "Presión sistólica", "glucosa_mg_dl": "Glucosa",
                 "paridad": "Paridad", "ca125_u_ml": "CA-125", "sexo / paridad": "Sexo y paridad"}
        rows = [[html.escape(str(r.control)), html.escape(names.get(str(r.variable), str(r.variable))), _miles(r.cantidad), _pct(r.porcentaje / 100, 1)] for r in issues.itertuples()]
        st.html(_table(["Control", "Variable", "Registros", "% de la selección"], rows, {2, 3}) if rows else '<p class="dk-note">Sin observaciones.</p>')
    with st.expander("Datos de la selección y descargas"):
        selected_id = st.text_input("Buscar identificador exacto de paciente", placeholder="Ejemplo: PAC-00001", key="stats_patient_id").strip()
        if selected_id:
            patient = data.loc[data.id_paciente.astype(str).str.casefold().eq(selected_id.casefold())]
            if patient.empty:
                st.info("Ese identificador no está en la selección actual.")
            else:
                st.html(record_table(patient.iloc[0]))
        a, b = st.columns(2)
        with a:
            st.download_button("Descargar pacientes de la selección", data.to_csv(index=False).encode("utf-8-sig"), "pacientes_seleccion.csv", "text/csv", key="stats_download_patients")
        with b:
            evidence = {"clasificacion": "Datos sintéticos", "fuente": metadata.get("file_name", ""), "sha256": metadata.get("sha256", metadata.get("hash", "")), "filtros": context}
            st.download_button("Descargar filtros y procedencia", json.dumps(evidence, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                               "contexto_analisis.json", "application/json", key="stats_download_context")
    with st.expander("Métodos"):
        st.markdown("- **Proporciones:** intervalo de Wilson del 95 %.\n"
                    "- **Factores asociados:** OR crudas con IC log-Wald del 95 %; no son efectos ajustados ni causales.\n"
                    "- **Modelo:** evaluado una sola vez en la prueba reservada (20 % de los pacientes); IC por bootstrap estratificado de 500 réplicas.\n"
                    "- **Alcance:** datos sintéticos; no establecen prevalencia hospitalaria ni riesgo clínico real.")


# --------------------------------------------------------------- Distribuciones
NUM_VARS = {"edad": ("Edad", "años"), "imc": ("IMC", "kg/m²"), "glucosa_mg_dl": ("Glucosa", "mg/dL"),
            "presion_sistolica": ("Presión sistólica", "mmHg"), "ca125_u_ml": ("CA-125", "U/mL"), "paridad": ("Paridad", "partos")}


def class_balance(data: pd.DataFrame) -> None:
    valid = data.riesgo_oncologico.isin([0, 1])
    n1 = int(data.loc[valid, "riesgo_oncologico"].sum())
    n0 = int(valid.sum()) - n1
    total = max(n0 + n1, 1)
    with st.container(key="dv_balance"):
        _card_head("Balance de clases", "Pacientes con etiqueta 0 y etiqueta 1 en la selección")
        st.html(f'''<div class="dv-bal"><i style="width:{n0 / total * 100:.1f}%;background:{GREEN2}"></i><i style="width:{n1 / total * 100:.1f}%;background:{CORAL}"></i></div>
          <div class="dv-bal-l"><span><b>{_pct(n0 / total)}</b> etiqueta 0 · {_miles(n0)}</span><span><b>{_pct(n1 / total)}</b> etiqueta 1 · {_miles(n1)}</span></div>
          <p class="dk-note">Hay un caso con etiqueta 1 por cada {n0 / max(n1, 1):.1f} sin ella; por eso se evalúan PR-AUC y la calibración, no solo la exactitud.</p>'''.replace(".", ",", 0))


def view_distributions(data: pd.DataFrame) -> None:
    options = list(NUM_VARS) + list(CAT_VARS)
    var = st.segmented_control("Variable", options, default="edad", key="stats_dist_var",
                               format_func=lambda v: (NUM_VARS.get(v) or CAT_VARS[v])[0], label_visibility="collapsed", bind="query-params")
    var = var or "edad"
    if var in CAT_VARS:
        categorical_view(data, var)
        return
    name, unit = NUM_VARS[var]
    d = data.loc[data.riesgo_oncologico.isin([0, 1]), [var, "riesgo_oncologico"]].dropna()
    g0, g1 = d.loc[d.riesgo_oncologico == 0, var], d.loc[d.riesgo_oncologico == 1, var]
    if g0.empty or g1.empty:
        st.info("La comparación necesita pacientes con etiqueta 0 y 1 en la selección.")
        return
    from scipy.stats import mannwhitneyu
    p = float(mannwhitneyu(g1, g0).pvalue)
    diff = g1.median() - g0.median()
    _msg(f"La mediana de {name.lower()} es <b>{_num(g1.median(), 1)} {unit}</b> con etiqueta 1 y <b>{_num(g0.median(), 1)} {unit}</b> con etiqueta 0 "
         f"({'+' if diff >= 0 else '−'}{_num(abs(diff), 1)}); {'la diferencia es estadísticamente clara' if p < .05 else 'no hay una diferencia clara'} (Mann–Whitney, p {'< 0,001' if p < .001 else '= ' + _num(p, 3)}).")
    left, right = st.columns([1.5, 1], gap="medium")
    with left, st.container(key="dv_hist"):
        _card_head(f"Distribución de {name.lower()}", "Porcentaje de pacientes de cada grupo en cada intervalo")
        edges = np.histogram_bin_edges(d[var], bins=24 if d[var].nunique() > 24 else d[var].nunique())
        mids = (edges[:-1] + edges[1:]) / 2
        h0 = np.histogram(g0, edges)[0] / len(g0) * 100
        h1 = np.histogram(g1, edges)[0] / len(g1) * 100
        hist = pd.DataFrame({"valor": np.round(mids, 1), "Etiqueta 0": np.round(h0, 2), "Etiqueta 1": np.round(h1, 2)})
        area = lambda color, alpha: {"type": "line", "smooth": True, "symbol": "none", "lineStyle": {"width": 2.5, "color": color},
                                     "areaStyle": {"color": color, "opacity": alpha}}
        st.echarts_chart({"grid": {"left": 4, "right": 10, "top": 30, "bottom": 26, "containLabel": True},
                          "legend": {"top": 0, "right": 0, "itemWidth": 10, "itemHeight": 10}, "tooltip": {"trigger": "axis"},
                          "color": [GREEN2, CORAL], "dataset": {"source": hist},
                          "xAxis": {"type": "category", "boundaryGap": False, "name": unit, "nameLocation": "middle", "nameGap": 26, **AXIS, "splitLine": {"show": False}},
                          "yAxis": {"type": "value", **AXIS, "axisLabel": {"color": "#9aa8a1", "formatter": "{value} %"}},
                          "series": [area(GREEN2, .18), area(CORAL, .16)]}, height=280, alt=f"Distribución de {name} por etiqueta")
    with right, st.container(key="dv_box"):
        _card_head("Diagrama de caja", "Mediana, cuartiles y bigotes a 1,5 × IQR")

        def five(s):
            q1, med, q3 = s.quantile([.25, .5, .75])
            iqr = q3 - q1
            return [round(float(max(s.min(), q1 - 1.5 * iqr)), 2), round(float(q1), 2), round(float(med), 2), round(float(q3), 2),
                    round(float(min(s.max(), q3 + 1.5 * iqr)), 2)]
        st.echarts_chart({"grid": {"left": 4, "right": 10, "top": 10, "bottom": 4, "containLabel": True}, "tooltip": {"trigger": "item"},
                          "xAxis": {"type": "category", "data": ["Etiqueta 0", "Etiqueta 1"], **AXIS, "splitLine": {"show": False}},
                          "yAxis": {"type": "value", "scale": True, **AXIS},
                          "series": [{"type": "boxplot", "boxWidth": [30, 56],
                                      "data": [{"value": five(g0), "itemStyle": {"color": "#e3f4ec", "borderColor": GREEN, "borderWidth": 2}},
                                               {"value": five(g1), "itemStyle": {"color": "#fde9e0", "borderColor": CORAL, "borderWidth": 2}}]}]},
                         height=200, alt=f"Diagrama de caja de {name} por etiqueta")
        rows = [[lab, _miles(len(s)), f"<b>{_num(s.mean(), 1)}</b>", _num(s.median(), 1), _num(s.std(), 1)] for lab, s in (("Etiqueta 0", g0), ("Etiqueta 1", g1))]
        st.html(_table(["Grupo", "n", "Media", "Mediana", "DE"], rows, {1, 2, 3, 4}))


# -------------------------------------------------------- piezas del modelo
FEATURE_NAMES = {"resultado_imagen_normal": "Imagen normal", "resultado_imagen_sospechoso": "Imagen sospechosa",
                 "resultado_imagen_indeterminado": "Imagen indeterminada", "antecedente_familiar_cancer_0": "Sin antecedente familiar",
                 "antecedente_familiar_cancer_1": "Antecedente familiar", "tabaquismo_nunca": "Nunca fumó", "tabaquismo_activo": "Tabaquismo activo",
                 "tabaquismo_exfumador": "Exfumador", "imc_categoria_bajo_peso": "Bajo peso", "imc_categoria_obesidad": "Obesidad",
                 "edad": "Edad", "ca125_u_ml": "CA-125", "antecedente_personal_benigno_0": "Sin antecedente benigno",
                 "antecedente_personal_benigno_1": "Antecedente benigno", "grupo_etario_<40": "Menor de 40 años"}
MODEL_NAMES = {"logistic_regression": "Regresión logística", "random_forest": "Random Forest", "xgboost": "XGBoost"}


def _summary() -> dict:
    from config import REPORTS_DIR
    try:
        return json.loads((REPORTS_DIR / "resumen_pipeline.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def model_comparison(summary: dict) -> None:
    oof = summary.get("desarrollo_fuera_de_pliegue", {})
    chosen = summary.get("modelo_seleccionado", "")
    if not oof:
        return
    with st.container(key="dv_compare"):
        _card_head("Comparación de modelos", "Validación cruzada agrupada por paciente sobre los datos de desarrollo (antes de tocar la prueba reservada)")
        frame = pd.DataFrame([{"modelo": MODEL_NAMES.get(k, k), "AUROC": v["modelo"]["auroc"], "PR-AUC": v["modelo"]["pr_auc"]} for k, v in oof.items()])
        left, right = st.columns([1, 1.1], gap="medium")
        with left:
            st.echarts_chart({"grid": {"left": 4, "right": 10, "top": 30, "bottom": 4, "containLabel": True},
                              "legend": {"top": 0, "right": 0, "itemWidth": 10, "itemHeight": 10}, "tooltip": {"trigger": "axis"},
                              "color": [GREEN, LIME], "dataset": {"source": frame}, "xAxis": {"type": "category", **AXIS, "splitLine": {"show": False}},
                              "yAxis": {"type": "value", "min": 0, "max": 1, **AXIS},
                              "series": [{"type": "bar", "itemStyle": {"color": GREEN, "borderRadius": [6, 6, 0, 0]}, "label": {"show": True, "position": "top", "formatter": "{@AUROC}", "fontSize": 10}},
                                         {"type": "bar", "itemStyle": {"color": LIME, "borderRadius": [6, 6, 0, 0]}, "label": {"show": True, "position": "top", "formatter": "{@[2]}", "fontSize": 10}}]},
                             height=230, alt="AUROC y PR-AUC de los tres modelos en validación cruzada")
        with right:
            rows = [[MODEL_NAMES.get(k, k) + (' <span class="dv-pill dn">elegido</span>' if k == chosen else ""),
                     _num(v["modelo"]["auroc"], 3), _num(v["modelo"]["pr_auc"], 3), _num(v["modelo"]["brier_score"], 3), _num(v["modelo"]["f1"], 3),
                     _pct(v["modelo"]["sensibilidad"], 0), _pct(v["modelo"]["especificidad"], 0)]
                    for k, v in oof.items()]
            null = next(iter(oof.values())).get("modelo_nulo", {})
            rows.append(['<span style="color:#7b8a83">Modelo nulo</span>', "0,500", _num(null.get("pr_auc", np.nan), 3), _num(null.get("brier_score", np.nan), 3), "—", "—", "—"])
            st.html(_table(["Modelo", "AUROC", "PR-AUC", "Brier", "F1", "Sens.", "Espec."], rows, {1, 2, 3, 4, 5, 6}) +
                    '<p class="dk-note">Se eligió el modelo con mejor AUROC y Brier; además es el más interpretable.</p>')


def model_contribution(summary: dict) -> None:
    items = summary.get("variables_mas_influyentes", [])
    if not items:
        return
    frame = pd.DataFrame([{"variable": FEATURE_NAMES.get(i["variable"].split("__", 1)[-1], i["variable"].split("__", 1)[-1]),
                           "coef": round(float(i["coeficiente (log-odds)"]), 3)} for i in items])
    frame = frame.reindex(frame.coef.abs().sort_values().index).tail(12).reset_index(drop=True)
    with st.container(key="dv_contrib"):
        _card_head("Variables con mayor contribución", "Coeficientes del modelo final (log-odds sobre variables estandarizadas)")
        st.echarts_chart({"grid": {"left": 4, "right": 40, "top": 6, "bottom": 4, "containLabel": True}, "tooltip": {"trigger": "item"},
                          "xAxis": {"type": "value", **AXIS}, "yAxis": {"type": "category", "data": frame.variable.tolist(), **AXIS,
                                                                       "axisLabel": {"color": INK, "fontSize": 12}, "splitLine": {"show": False}},
                          "series": [{"type": "bar", "barWidth": "60%",
                                      "data": [{"value": c, "itemStyle": {"color": CORAL if c > 0 else GREEN2, "borderRadius": [0, 6, 6, 0] if c > 0 else [6, 0, 0, 6]}} for c in frame.coef],
                                      "label": {"show": True, "position": "right", "formatter": "{c}", "color": INK, "fontSize": 11}}]},
                         height=330, alt="Coeficientes de las variables con mayor contribución en el modelo")
        st.html('<p class="dk-note">Naranja aumenta la probabilidad estimada; verde la disminuye. Es aporte predictivo dentro del modelo, no causalidad.</p>')


def model_subgroups(holdout: pd.DataFrame, data_full: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score
    merged = holdout.merge(data_full[["id_paciente", "sexo", "edad"]], on="id_paciente", how="left")
    merged["grupo"] = pd.cut(merged.edad, [0, 40, 55, 70, 200], right=False, labels=["<40", "40–54", "55–69", "70+"])
    rows = []
    for title, col, mapping in (("Sexo", "sexo", {"F": "Mujeres", "M": "Hombres"}), ("Edad", "grupo", None)):
        for key, part in merged.groupby(col, observed=True):
            if part.etiqueta.nunique() == 2 and len(part) >= 30:
                rows.append({"grupo": f"{title}: {mapping.get(key, key) if mapping else key}", "auroc": round(float(roc_auc_score(part.etiqueta, part.probabilidad)), 3), "n": len(part)})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    with st.container(key="dv_sub"):
        _card_head("Estabilidad por subgrupos", "AUROC en la prueba reservada para cada grupo")
        st.echarts_chart({"grid": {"left": 4, "right": 40, "top": 6, "bottom": 4, "containLabel": True}, "tooltip": {"trigger": "axis"},
                          "dataset": {"source": frame}, "xAxis": {"type": "value", "min": .5, "max": .85, **AXIS},
                          "yAxis": {"type": "category", "inverse": True, **AXIS, "axisLabel": {"color": INK}, "splitLine": {"show": False}},
                          "series": [{"type": "bar", "encode": {"x": "auroc", "y": "grupo"}, "barWidth": "55%",
                                      "itemStyle": {"color": GREEN2, "borderRadius": [0, 6, 6, 0]},
                                      "label": {"show": True, "position": "right", "formatter": "{@auroc}", "color": INK}}]},
                         height=260, alt="AUROC del modelo por sexo y grupo de edad")
        st.html('<p class="dk-note">Si las barras son parecidas, el modelo funciona de forma similar en cada grupo. Con grupos pequeños la estimación es menos precisa.</p>')


def pr_curve(results: dict, prevalence: float) -> None:
    pr = results["pr"].rename(columns={"Recall": "recall", "Precision": "precision"})
    if len(pr) > 400:
        pr = pr.iloc[np.linspace(0, len(pr) - 1, 400).astype(int)]
    with st.container(key="dv_pr"):
        _card_head("Curva precisión–sensibilidad (PR)", "Útil cuando la etiqueta 1 es minoritaria")
        st.echarts_chart({"grid": {"left": 4, "right": 14, "top": 10, "bottom": 26, "containLabel": True}, "tooltip": {"trigger": "axis"},
                          "dataset": {"source": pr.reset_index(drop=True)},
                          "xAxis": {"type": "value", "min": 0, "max": 1, "name": "Sensibilidad", "nameLocation": "middle", "nameGap": 26, **AXIS},
                          "yAxis": {"type": "value", "min": 0, "max": 1, **AXIS},
                          "series": [{"type": "line", "symbol": "none", "encode": {"x": "recall", "y": "precision"}, "lineStyle": {"width": 3, "color": GREEN},
                                      "areaStyle": {"color": "rgba(31,185,133,.12)"}},
                                     {"type": "line", "symbol": "none", "data": [[0, round(prevalence, 3)], [1, round(prevalence, 3)]],
                                      "lineStyle": {"type": "dashed", "color": "#c5ccd8"}}]},
                         height=260, alt="Curva de precisión y sensibilidad del modelo")


# ------------------------------------------------------------- valores atípicos
def outliers_chart(report: dict) -> None:
    cols = report["columns"]
    frame = cols.loc[cols.variable.isin(list(NUM_VARS)), ["variable", "atipicos_iqr"]].copy()
    frame["variable"] = frame.variable.map(lambda v: NUM_VARS[v][0])
    frame = frame.sort_values("atipicos_iqr").reset_index(drop=True)
    with st.container(key="dv_outliers"):
        _card_head("Valores atípicos por variable", "Pacientes fuera de Q1 − 1,5 × IQR o Q3 + 1,5 × IQR; se revisan, no se borran")
        st.echarts_chart({"grid": {"left": 4, "right": 50, "top": 6, "bottom": 4, "containLabel": True}, "tooltip": {"trigger": "axis"},
                          "dataset": {"source": frame}, "xAxis": {"type": "value", **AXIS},
                          "yAxis": {"type": "category", **AXIS, "axisLabel": {"color": INK}, "splitLine": {"show": False}},
                          "series": [{"type": "bar", "encode": {"x": "atipicos_iqr", "y": "variable"}, "barWidth": "55%",
                                      "itemStyle": {"color": LIME, "borderRadius": [0, 6, 6, 0]},
                                      "label": {"show": True, "position": "right", "formatter": "{@atipicos_iqr}", "color": INK}}]},
                         height=240, alt="Cantidad de valores atípicos por variable numérica")


# ------------------------------------------------------------ categóricas
CAT_VARS = {"sexo": ("Sexo", {"F": "Femenino", "M": "Masculino"}),
            "antecedente_familiar_cancer": ("Antecedente familiar", {0: "No", 1: "Sí"}),
            "antecedente_personal_benigno": ("Antecedente benigno", {0: "No", 1: "Sí"}),
            "tabaquismo": ("Tabaquismo", {"nunca": "Nunca", "exfumador": "Exfumador", "activo": "Activo"}),
            "resultado_imagen": ("Hallazgo de imagen", {"normal": "Normal", "indeterminado": "Indeterminado", "sospechoso": "Sospechoso"}),
            "imc_categoria": ("Categoría de IMC", None)}


def categorical_view(data: pd.DataFrame, var: str) -> None:
    name, mapping = CAT_VARS[var]
    d = data.loc[data.riesgo_oncologico.isin([0, 1]), [var, "riesgo_oncologico"]].dropna()
    if d.empty:
        st.info("No hay pacientes con etiqueta válida en la selección.")
        return
    lab = (lambda v: mapping.get(v, str(v))) if mapping else (lambda v: str(v).replace("_", " ").capitalize())
    grouped = d.groupby(var, observed=True)["riesgo_oncologico"].agg(pacientes="size", positivos="sum").reset_index()
    grouped["categoria"] = grouped[var].map(lab)
    grouped["tasa"] = (grouped.positivos / grouped.pacientes * 100).round(1)
    grouped["porcentaje"] = (grouped.pacientes / grouped.pacientes.sum() * 100).round(1)
    top = grouped.sort_values("tasa").iloc[-1]
    low = grouped.sort_values("tasa").iloc[0]
    _msg(f"Con {name.lower()} <b>{top.categoria.lower()}</b>, el {_num(top.tasa, 1)} % tiene etiqueta 1, frente al "
         f"{_num(low.tasa, 1)} % con <b>{low.categoria.lower()}</b>. Es la distribución según la condición registrada, no una causa.")
    left, right = st.columns([1, 1.2], gap="medium")
    with left, st.container(key="dv_catfreq"):
        _card_head(f"Frecuencia de {name.lower()}", "Pacientes de la selección en cada categoría")
        st.echarts_chart({"grid": {"left": 4, "right": 50, "top": 6, "bottom": 4, "containLabel": True}, "tooltip": {"trigger": "axis"},
                          "dataset": {"source": grouped[["categoria", "pacientes", "porcentaje"]]},
                          "xAxis": {"type": "value", **AXIS}, "yAxis": {"type": "category", **AXIS, "axisLabel": {"color": INK}, "splitLine": {"show": False}},
                          "series": [{"type": "bar", "encode": {"x": "pacientes", "y": "categoria"}, "barWidth": "55%",
                                      "itemStyle": {"color": GREEN2, "borderRadius": [0, 6, 6, 0]},
                                      "label": {"show": True, "position": "right", "formatter": "{@porcentaje} %", "color": INK}}]},
                         height=240, alt=f"Pacientes por categoría de {name}")
    with right, st.container(key="dv_catrate"):
        _card_head(f"Etiqueta 1 según {name.lower()}", "Porcentaje con etiqueta 1 dentro de cada categoría")
        stack = grouped.assign(**{"Etiqueta 1": grouped.tasa, "Etiqueta 0": (100 - grouped.tasa).round(1)})[["categoria", "Etiqueta 1", "Etiqueta 0"]]
        st.echarts_chart({"grid": {"left": 4, "right": 10, "top": 30, "bottom": 4, "containLabel": True},
                          "legend": {"top": 0, "right": 0, "itemWidth": 10, "itemHeight": 10}, "tooltip": {"trigger": "axis"},
                          "color": [CORAL, "#dcebe4"], "dataset": {"source": stack},
                          "xAxis": {"type": "category", **AXIS, "splitLine": {"show": False}},
                          "yAxis": {"type": "value", "max": 100, **AXIS, "axisLabel": {"color": "#9aa8a1", "formatter": "{value} %"}},
                          "series": [{"type": "bar", "stack": "t", "barWidth": "50%", "label": {"show": True, "formatter": "{@[1]} %", "color": INK, "fontWeight": 700}},
                                     {"type": "bar", "stack": "t", "itemStyle": {"borderRadius": [6, 6, 0, 0]}}]},
                         height=240, alt=f"Proporción de etiqueta 1 según {name}")
        rows = [[html.escape(r.categoria), _miles(r.pacientes), _miles(r.positivos), f"<b>{_num(r.tasa, 1)} %</b>"] for r in grouped.itertuples()]
        st.html(_table(["Categoría", "Pacientes", "Etiqueta 1", "%"], rows, {1, 2, 3}))


def missing_chart(report: dict) -> None:
    cols = report["columns"].copy()
    names = {"id_paciente": "Identificador", "edad": "Edad", "sexo": "Sexo", "antecedente_familiar_cancer": "Antecedente familiar",
             "antecedente_personal_benigno": "Antecedente benigno", "tabaquismo": "Tabaquismo", "imc": "IMC", "presion_sistolica": "Presión sistólica",
             "glucosa_mg_dl": "Glucosa", "paridad": "Paridad", "ca125_u_ml": "CA-125", "resultado_imagen": "Hallazgo de imagen",
             "fuente": "Fuente", "periodo": "Semestre", "riesgo_oncologico": "Etiqueta"}
    cols = cols.loc[cols.variable.isin(names)].copy()
    cols["variable"] = cols.variable.map(names)
    cols["ausentes_pct"] = cols.ausentes_pct.astype(float).round(2)
    cols = cols.sort_values("ausentes_pct").reset_index(drop=True)
    with st.container(key="dv_missing"):
        total = int(cols.ausentes.sum())
        _card_head("Valores faltantes por variable", "Porcentaje de celdas vacías en cada variable" + (" · ninguna variable tiene faltantes" if total == 0 else ""))
        st.echarts_chart({"grid": {"left": 4, "right": 40, "top": 6, "bottom": 4, "containLabel": True}, "tooltip": {"trigger": "axis"},
                          "dataset": {"source": cols[["variable", "ausentes_pct"]]},
                          "xAxis": {"type": "value", "min": 0, "max": max(5, float(cols.ausentes_pct.max()) * 1.2), **AXIS,
                                    "axisLabel": {"color": "#9aa8a1", "formatter": "{value} %"}},
                          "yAxis": {"type": "category", **AXIS, "axisLabel": {"color": INK, "fontSize": 11}, "splitLine": {"show": False}},
                          "series": [{"type": "bar", "encode": {"x": "ausentes_pct", "y": "variable"}, "barWidth": "55%", "barMinHeight": 2,
                                      "itemStyle": {"color": CORAL, "borderRadius": [0, 6, 6, 0]},
                                      "label": {"show": True, "position": "right", "formatter": "{@ausentes_pct} %", "color": MUTE, "fontSize": 10}}]},
                         height=300, alt="Porcentaje de valores faltantes por variable")
