"""Exploración estadística reproducible de la cohorte sintética de OncoRisk."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats

from src.analytics.data import load_patient_data
from src.analytics import statistics as stats
from src.analytics.model_diagnostics import recover_holdout, diagnostic_results
from config import ANALYTICAL_DIR
from app.dashboard_cards import render_dashboard
from app.detail_views import VIEWS, view_cohort, view_distributions, view_factors, view_model, view_quality

NUMERIC = ["edad", "imc", "presion_sistolica", "glucosa_mg_dl", "paridad", "ca125_u_ml"]
LABELS = {
    "id_paciente": "Paciente sintético", "edad": "Edad (años)", "sexo": "Sexo",
    "imc": "IMC (kg/m²)", "presion_sistolica": "Presión sistólica (mmHg)",
    "glucosa_mg_dl": "Glucosa (mg/dL)", "paridad": "Paridad", "ca125_u_ml": "CA-125 (U/mL)",
    "antecedente_familiar_cancer": "Antecedente familiar de cáncer",
    "antecedente_personal_benigno": "Antecedente personal benigno",
    "tabaquismo": "Tabaquismo", "resultado_imagen": "Hallazgo de imagen",
    "fuente": "Fuente", "periodo": "Semestre", "riesgo_oncologico": "Etiqueta simulada",
    "grupo_etario": "Grupo de edad", "imc_categoria": "Categoría de IMC",
}
VARIABLE_LABELS = LABELS
LABELS["sexo / paridad"] = "Sexo y paridad"
LABELS["todas"] = "Todas las variables"
RESULT_LABELS = {
    "variable": "Variable", "variable_x": "Variable X", "variable_y": "Variable Y",
    "n": "Datos válidos", "n_total": "Registros", "n_prueba": "Tamaño de la prueba",
    "ausentes": "Ausentes", "media": "Media", "mediana": "Mediana", "desv_est": "Desviación estándar",
    "varianza": "Varianza", "minimo": "Mínimo", "maximo": "Máximo", "q1": "Primer cuartil",
    "q3": "Tercer cuartil", "iqr": "IQR", "cv_pct": "CV (%)", "asimetria": "Asimetría",
    "curtosis_exceso": "Curtosis en exceso", "ic_inf": "IC 95% inferior", "ic_sup": "IC 95% superior",
    "n0": "Datos de etiqueta 0", "n1": "Datos de etiqueta 1", "media0": "Media: etiqueta 0",
    "media1": "Media: etiqueta 1", "mediana0": "Mediana: etiqueta 0", "mediana1": "Mediana: etiqueta 1",
    "diferencia_media": "Diferencia de medias (1−0)", "t_welch": "Estadístico t de Welch",
    "p_welch": "p de Welch", "q_welch": "q de Welch (BH)", "u_mannwhitney": "U de Mann–Whitney",
    "p_mannwhitney": "p de Mann–Whitney", "q_mannwhitney": "q de Mann–Whitney (BH)",
    "hedges_g": "g de Hedges", "rank_biserial": "Efecto biserial de rangos", "estado": "Observación",
    "p": "Valor p", "q": "Valor q (BH)", "w": "W de Shapiro–Wilk", "metodo": "Método",
    "coeficiente": "Correlación", "cramers_v": "V de Cramér", "estadistico": "Estadístico",
    "min_esperado": "Mínimo esperado", "fraccion_esperada_menor5": "Fracción de esperados < 5",
    "grupo": "Grupo", "categoria": "Categoría", "referencia": "Referencia", "n_categoria": "Datos de categoría",
    "n_referencia": "Datos de referencia", "or": "Odds Ratio", "p_fisher": "p de Fisher",
    "correccion": "Corrección de ceros", "porcentaje": "Porcentaje (%)", "positivos": "Etiqueta 1",
    "tipo": "Tipo de dato", "ausentes_pct": "Ausentes (%)", "unicos": "Valores distintos",
    "invalidos": "Fuera de dominio", "atipicos_iqr": "Atípicos por IQR", "control": "Control",
    "cantidad": "Cantidad", "detalle": "Detalle", "cluster": "Grupo exploratorio", "proporcion_pct": "Composición (%)",
    "n_etiqueta": "Etiquetas válidas", "etiqueta1_pct": "Etiqueta 1 (%)", "etiqueta": "Etiqueta simulada",
    "componente": "Componente", "varianza_pct": "Varianza explicada (%)", "acumulada_pct": "Varianza acumulada (%)",
    "PC1": "Componente principal 1", "PC2": "Componente principal 2", "PC3": "Componente principal 3",
    "PC4": "Componente principal 4", "PC5": "Componente principal 5", "PC6": "Componente principal 6",
    "pacientes": "Pacientes", "valor": "Valor", "densidad": "Densidad", "normal": "Cuantil normal",
    "observado": "Cuantil observado", "acumulada": "Acumulado (%)", "desde": "Desde", "hasta": "Hasta",
    "desviacion": "Desviación estándar", "n_movil": "Datos de ventana móvil", "media_movil": "Media móvil ponderada",
    "movil_inf": "IC móvil inferior", "movil_sup": "IC móvil superior",
}
GROUPS = ["sexo", "grupo_etario", "imc_categoria", "tabaquismo", "resultado_imagen",
          "antecedente_familiar_cancer", "antecedente_personal_benigno", "fuente"]
COLORS = ["#14946a", "#ff8a5c", "#b5dd4a", "#5b8def", "#e3a008", "#7b8a83"]
CONFIG = {"view": {"stroke": None}, "font": "Instrument Sans", "background": "transparent",
          "axis": {"labelColor": "#6d6f76", "titleColor": "#6d6f76", "gridColor": "#e6ede9", "domainColor": "#c9c6bf", "tickColor": "#c9c6bf", "labelFontSize": 11, "titleFontWeight": 500},
          "legend": {"labelColor": "#0d0f14", "titleColor": "#6d6f76", "orient": "top"}, "bar": {"cornerRadiusEnd": 6}}


def label(column: str) -> str:
    return LABELS.get(column, RESULT_LABELS.get(column, column.replace("_", " ").capitalize() if "_" in column else column))


def category(value: Any, column: str) -> str:
    if pd.isna(value):
        return "Sin dato"
    if column == "riesgo_oncologico":
        return f"Etiqueta {int(float(value))}" if str(value) in ("0", "1", "0.0", "1.0") else str(value)
    if column in ("antecedente_familiar_cancer", "antecedente_personal_benigno"):
        return {"0": "No", "1": "Sí", "0.0": "No", "1.0": "Sí"}.get(str(value), str(value))
    return {"F": "Femenino", "M": "Masculino", "consulta_externa": "Consulta externa",
            "remision": "Remisión", "urgencias": "Urgencias", "nunca": "Nunca",
            "exfumador": "Exfumador", "activo": "Activo", "normal": "Normal",
            "indeterminado": "Indeterminado", "sospechoso": "Sospechoso",
            "bajo_peso": "Bajo peso", "normal_imc": "Normal", "sobrepeso": "Sobrepeso",
            "obesidad": "Obesidad"}.get(str(value), str(value))


def show_chart(data: pd.DataFrame, spec: dict, height: int = 300):
    if data.empty:
        st.info("No hay observaciones válidas para esta gráfica.")
        return
    def translated(node):
        if isinstance(node, list):
            return [translated(item) for item in node]
        if not isinstance(node, dict):
            return node
        result = {key: translated(value) for key, value in node.items()}
        for secondary in ("x2", "y2", "xOffset", "yOffset"):
            if isinstance(result.get(secondary), dict):
                result[secondary].pop("title", None)
        if "field" in result and "title" not in result:
            result["title"] = label(result["field"])
        return result
    st.vega_lite_chart(data, {"height": height, "config": CONFIG, **translated(spec)}, width="stretch")


def frame_csv(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8-sig")


def table(data: pd.DataFrame):
    display = readable(data).rename(columns=label)
    configuration = {}
    for column in display.select_dtypes(include="number"):
        if column.startswith("p ") or column.startswith("q ") or column in ("Valor p", "Valor q (BH)"):
            configuration[column] = st.column_config.NumberColumn(column, format="%.3e")
    st.dataframe(display, hide_index=True, width="stretch", column_config=configuration)


def readable(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    for field in ("variable", "variable_x", "variable_y"):
        if field in output:
            output[field] = output[field].map(label)
    if "variable" in data:
        for field in ("categoria", "referencia"):
            if field in data:
                output[field] = [category(v, c) for v, c in zip(data[field], data["variable"])]
    return output


def bar_frequency(data: pd.DataFrame, column: str, title: str):
    st.markdown(f"#### {title}")
    frequency = stats.frequency_table(data, column).copy()
    frequency["categoria"] = frequency["categoria"].map(lambda x: category(x, column))
    show_chart(frequency, {"mark": {"type": "bar", "color": COLORS[0]}, "encoding": {
        "y": {"field": "categoria", "type": "nominal", "title": None, "sort": "-x", "axis": {"labelLimit": 200}},
        "x": {"field": "n", "type": "quantitative", "title": "Pacientes", "scale": {"zero": True}},
        "tooltip": [{"field": "categoria", "type": "nominal", "title": label(column)},
                    {"field": "n", "type": "quantitative", "title": "Pacientes", "format": ",.0f"},
                    {"field": "porcentaje", "type": "quantitative", "title": "% de la selección", "format": ".2f"}],
    }}, height=max(180, len(frequency) * 37))


def rate_chart(data: pd.DataFrame, group: str):
    rates = stats.group_rates(data, group).copy()
    rates["grupo"] = rates[group].map(lambda x: category(x, group))
    show_chart(rates, {"layer": [
        {"mark": {"type": "rule", "color": COLORS[0], "strokeWidth": 2}, "encoding": {
            "x": {"field": "ic_inf", "type": "quantitative", "scale": {"domain": [0, 100]}, "title": "Etiqueta 1 (%) · IC del 95% de Wilson"},
            "x2": {"field": "ic_sup"}, "y": {"field": "grupo", "type": "nominal", "title": None, "sort": None}}},
        {"mark": {"type": "point", "filled": True, "size": 90, "color": COLORS[0]}, "encoding": {
            "x": {"field": "porcentaje", "type": "quantitative", "scale": {"domain": [0, 100]}},
            "y": {"field": "grupo", "type": "nominal", "title": None, "sort": None},
            "tooltip": [{"field": "grupo", "type": "nominal", "title": label(group)},
                        {"field": "n", "type": "quantitative", "title": "Etiquetas válidas"},
                        {"field": "positivos", "type": "quantitative", "title": "Etiqueta 1"},
                        {"field": "porcentaje", "type": "quantitative", "title": "% etiqueta 1", "format": ".2f"},
                        {"field": "ic_inf", "type": "quantitative", "title": "IC inferior (%)", "format": ".2f"},
                        {"field": "ic_sup", "type": "quantitative", "title": "IC superior (%)", "format": ".2f"}]}}
    ]}, height=max(230, len(rates) * 40))
    st.caption("Cada porcentaje usa como denominador las etiquetas 0/1 válidas de su propio grupo. Los intervalos describen incertidumbre de muestreo bajo un modelo binomial; no validan las etiquetas sintéticas.")
    with st.expander("Ver denominadores e intervalos por grupo"):
        table(rates.drop(columns=[group]).rename(columns={"grupo": label(group)}).round(3))


def _reset_filters(defaults: dict):
    for name, value in defaults.items():
        st.session_state[name] = value


def apply_filters(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ages = pd.to_numeric(source["edad"], errors="coerce").dropna()
    age_range = (float(ages.min()), float(ages.max())) if not ages.empty else (0.0, 100.0)
    defaults = {"stats_age": age_range}
    for column in ("sexo", "riesgo_oncologico", "fuente", "periodo"):
        defaults[f"stats_{column}"] = sorted(source[column].dropna().unique().tolist())
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    with st.container(key="stats_filters"):
        st.html('<div class="or-filters-h"><h3>Filtrar la cohorte</h3><p>Los filtros aplican al resumen y a todas las vistas de pacientes.</p></div>')
        first, second, third = st.columns([2, 1, 1])
        with first:
            if age_range[0] < age_range[1]:
                selected_age = st.slider("Rango de edad (años)", min_value=age_range[0], max_value=age_range[1], step=0.1, key="stats_age")
            else:
                selected_age = age_range
                st.caption(f"Edad disponible: {age_range[0]:g} años")
        with second:
            selected_sex = st.multiselect("Sexo", defaults["stats_sexo"], format_func=lambda x: category(x, "sexo"), key="stats_sexo")
        with third:
            selected_target = st.multiselect("Etiqueta simulada", defaults["stats_riesgo_oncologico"], format_func=lambda x: category(x, "riesgo_oncologico"), key="stats_riesgo_oncologico")
        left, middle, right = st.columns([2, 2, 1])
        with left:
            selected_source = st.multiselect("Fuente de procedencia", defaults["stats_fuente"], format_func=lambda x: category(x, "fuente"), key="stats_fuente")
        with middle:
            selected_period = st.multiselect("Semestre de procedencia", defaults["stats_periodo"], key="stats_periodo")
        with right:
            st.button("Restablecer filtros", on_click=_reset_filters, args=(defaults,), key="stats_reset", width="stretch")
    filtered = source.loc[source.edad.between(*selected_age) & source.sexo.isin(selected_sex)
                          & source.riesgo_oncologico.isin(selected_target) & source.fuente.isin(selected_source)
                          & source.periodo.isin(selected_period)].copy()
    context = {"edad_min": selected_age[0], "edad_max": selected_age[1], "sexo": selected_sex,
               "etiquetas": [int(x) for x in selected_target], "fuentes": selected_source, "semestres": selected_period,
               "filas": len(filtered), "pacientes_unicos": int(filtered.id_paciente.nunique())}
    share = len(filtered) / len(source) if len(source) else 0
    st.html(f'<div class="or-cohort"><b>{len(filtered):,}</b> de {len(source):,} pacientes seleccionados<span class="or-cohort-bar"><i style="width:{share * 100:.1f}%"></i></span><small>La vista «Evaluación del modelo» usa por separado su prueba reservada original.</small></div>'.replace(",", "."))
    return filtered, context


def overview(data: pd.DataFrame):
    valid = data.riesgo_oncologico.isin([0, 1])
    denominator = int(valid.sum())
    positives = int(data.loc[valid, "riesgo_oncologico"].sum())
    proportion = positives / denominator if denominator else np.nan
    low, high = stats.wilson_interval(positives, denominator)
    source_columns = [c for c in data if c not in ("grupo_etario", "imc_categoria")]
    completeness = 100 * (1 - data[source_columns].isna().sum().sum() / (len(data) * len(source_columns)))
    a, b, c, d = st.columns(4)
    a.metric("Pacientes en la selección", f"{data.id_paciente.nunique():,}".replace(",", "."))
    b.metric("Etiqueta 1", _pct(proportion, 2) if denominator else "Sin dato")
    c.metric("Edad promedio", f"{data.edad.mean():.1f}".replace(".", ",") + " años")
    d.metric("Completitud de celdas", f"{completeness:.2f}".replace(".", ",") + " %")
    st.caption(f"Etiqueta 1: {positives:,} / {denominator:,} etiquetas válidas · IC del 95% de Wilson: {low:.2%}–{high:.2%}. El identificador es ficticio y la etiqueta es simulada.")
    left, right = st.columns(2)
    with left:
        bar_frequency(data, "sexo", "Composición por sexo")
        bar_frequency(data, "riesgo_oncologico", "Distribución de la etiqueta simulada")
    with right:
        bar_frequency(data, "grupo_etario", "Composición por edad")
        bar_frequency(data, "fuente", "Procedencia de los registros")
    st.markdown("#### Sexo y grupos de edad")
    demographic = data.groupby(["grupo_etario", "sexo"], observed=True).size().rename("pacientes").reset_index()
    demographic["sexo"] = demographic.sexo.map(lambda value: category(value, "sexo"))
    show_chart(demographic, {"mark": {"type": "bar", }, "encoding": {
        "x": {"field": "grupo_etario", "type": "ordinal", "sort": ["<40", "40-54", "55-69", "70+"], "axis": {"labelAngle": 0}},
        "xOffset": {"field": "sexo"}, "y": {"field": "pacientes", "type": "quantitative"},
        "color": {"field": "sexo", "type": "nominal", "scale": {"range": COLORS[:2]}},
        "tooltip": [{"field": "grupo_etario"}, {"field": "sexo"}, {"field": "pacientes", "type": "quantitative"}]}})
    st.caption("Los grupos de edad corresponden a intervalos [18,40), [40,55), [55,70) y [70,100]. Las edades pueden contener decimales.")


def distributions(data: pd.DataFrame, summary: pd.DataFrame):
    variable, group = st.columns(2)
    with variable:
        column = st.selectbox("Variable numérica", NUMERIC, format_func=label, key="stats_distribution_variable")
    with group:
        grouping = st.selectbox("Comparar distribución por", ["riesgo_oncologico", "sexo", "tabaquismo", "resultado_imagen"], format_func=label, key="stats_distribution_group")
    values = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    bins = st.slider("Número de intervalos del histograma", 8, 60, 24, key="stats_histogram_bins")
    counts, edges = np.histogram(values.to_numpy(), bins=bins) if len(values) else (np.array([]), np.array([]))
    hist = pd.DataFrame({"desde": edges[:-1], "hasta": edges[1:], "pacientes": counts})
    left, right = st.columns(2)
    with left:
        st.markdown("#### Histograma")
        show_chart(hist, {"mark": {"type": "bar", "color": COLORS[0]}, "encoding": {
            "x": {"field": "desde", "type": "quantitative", "bin": "binned", "title": label(column)},
            "x2": {"field": "hasta"}, "y": {"field": "pacientes", "type": "quantitative", "title": "Pacientes"},
            "tooltip": [{"field": "desde", "type": "quantitative", "title": "Desde", "format": ".2f"}, {"field": "hasta", "type": "quantitative", "title": "Hasta", "format": ".2f"}, {"field": "pacientes", "type": "quantitative"}]}})
    with right:
        st.markdown("#### Distribución acumulada empírica")
        ecdf = values.sort_values().value_counts(sort=False).sort_index().rename_axis("valor").reset_index(name="n")
        ecdf["acumulada"] = 100 * ecdf.n.cumsum() / len(values) if len(values) else []
        show_chart(ecdf, {"mark": {"type": "line", "interpolate": "step-after", "color": COLORS[0]}, "encoding": {
            "x": {"field": "valor", "type": "quantitative", "title": label(column)},
            "y": {"field": "acumulada", "type": "quantitative", "title": "Observaciones acumuladas (%)", "scale": {"domain": [0, 100]}},
            "tooltip": [{"field": "valor", "type": "quantitative", "title": label(column)}, {"field": "acumulada", "type": "quantitative", "title": "Acumulado (%)", "format": ".2f"}]}})
    left, right = st.columns(2)
    with left:
        st.markdown("#### Mediana y dispersión por grupo")
        boxes = data[[column, grouping]].dropna().copy()
        boxes["grupo"] = boxes[grouping].map(lambda x: category(x, grouping))
        show_chart(boxes, {"mark": {"type": "boxplot", "extent": 1.5, "outliers": False, "size": 35, "color": COLORS[0]}, "encoding": {
            "x": {"field": "grupo", "type": "nominal", "title": label(grouping), "axis": {"labelAngle": -20}},
            "y": {"field": column, "type": "quantitative", "title": label(column), "scale": {"zero": False}}}})
        st.caption("Caja: Q1–Q3; línea: mediana; bigotes: hasta 1,5 × IQR. Los puntos atípicos se cuentan en Calidad y datos.")
    with right:
        st.markdown("#### Gráfica Q–Q frente a una normal")
        if len(values) >= 3 and values.std(ddof=1) > 0:
            probability = np.linspace(0.005, 0.995, min(199, len(values)))
            qq = pd.DataFrame({"normal": scipy_stats.norm.ppf(probability), "observado": (values.quantile(probability).to_numpy() - values.mean()) / values.std(ddof=1)})
            show_chart(qq, {"layer": [{"mark": {"type": "point", "color": COLORS[0], "opacity": .7}, "encoding": {
                "x": {"field": "normal", "type": "quantitative", "title": "Cuantil normal teórico"},
                "y": {"field": "observado", "type": "quantitative", "title": "Cuantil observado estandarizado"},
                "tooltip": [{"field": "normal", "type": "quantitative"}, {"field": "observado", "type": "quantitative"}]}},
                {"mark": {"type": "line", "color": "#de8a50", "strokeDash": [5, 4]}, "encoding": {"x": {"field": "normal", "type": "quantitative"}, "y": {"field": "normal", "type": "quantitative"}}}]})
        else:
            st.info("Se necesitan al menos tres valores y variación para esta gráfica.")
        st.caption("Las desviaciones respecto de la diagonal muestran diferencias de forma frente a una normal. Es un diagnóstico visual, no una prueba de normalidad.")
    st.markdown("#### Estadísticos descriptivos de las seis variables")
    table(readable(summary).round(4))
    st.caption("Desviación y varianza muestrales (n−1); curtosis en exceso; CV = 100 × desviación / |media|. IC del 95% de la media mediante t de Student, bajo independencia y aproximación muestral. Las cifras proceden de una cohorte artificial.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Densidad suavizada (KDE)")
        density = stats.density_curve(values)
        show_chart(density, {"mark": {"type": "area", "opacity": .45, "line": {"color": COLORS[0]}, "color": COLORS[0]}, "encoding": {
            "x": {"field": "valor", "type": "quantitative", "title": label(column), "scale": {"zero": False}},
            "y": {"field": "densidad", "type": "quantitative", "title": "Densidad estimada"},
            "tooltip": [{"field": "valor", "type": "quantitative", "title": label(column)}, {"field": "densidad", "type": "quantitative", "format": ".4f"}]}})
        st.caption("KDE gaussiana, ancho de banda de Scott, con todos los valores válidos. Su altura es densidad, no número de pacientes. Para paridad discreta, prioriza las frecuencias y el histograma.")
    with right:
        st.markdown("#### Violines por grupo")
        curves = []
        for value, group in data.groupby(grouping, observed=True):
            curve = stats.density_curve(group[column], points=80)
            curve["grupo"] = category(value, grouping)
            curve["izquierda"] = -curve.densidad
            curve["derecha"] = curve.densidad
            curves.append(curve)
        violin = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
        show_chart(violin, {"facet": {"column": {"field": "grupo", "type": "nominal", "title": label(grouping)}},
            "spec": {"width": 90, "height": 270, "mark": {"type": "area", "orient": "horizontal", "color": COLORS[0], "opacity": .65}, "encoding": {
                "y": {"field": "valor", "type": "quantitative", "title": label(column), "scale": {"zero": False}},
                "x": {"field": "izquierda", "type": "quantitative", "title": None, "axis": None}, "x2": {"field": "derecha"},
                "tooltip": [{"field": "grupo", "title": label(grouping)}, {"field": "valor", "type": "quantitative", "title": label(column)}, {"field": "densidad", "type": "quantitative", "format": ".4f"}]}},
            "resolve": {"scale": {"y": "shared", "x": "shared"}}})
        st.caption("Ancho: densidad relativa dentro de cada grupo; no representa el tamaño del grupo. Grupos constantes o con menos de 3 datos no tienen KDE.")
    st.markdown("#### Pruebas de normalidad de Shapiro–Wilk")
    normality = stats.normality_table(data)
    table(normality)
    st.caption("Hipótesis nula: distribución normal. Para evitar el límite de Shapiro con n > 5.000, se usa una submuestra reproducible de hasta 5.000 por variable (semilla 42). q ajusta las seis pruebas con Benjamini–Hochberg. Un p pequeño no mide la magnitud de la desviación ni elige automáticamente el contraste entre grupos.")
    st.download_button("Descargar estadísticos descriptivos", frame_csv(readable(summary)), "estadisticos_descriptivos.csv", "text/csv", key="stats_download_summary")
    st.download_button("Descargar pruebas de normalidad", frame_csv(readable(normality)), "normalidad.csv", "text/csv", key="stats_download_normality")


def relationships(data: pd.DataFrame):
    a, b, c = st.columns(3)
    with a:
        x = st.selectbox("Variable del eje X", NUMERIC, format_func=label, key="stats_scatter_x")
    with b:
        y = st.selectbox("Variable del eje Y", NUMERIC, index=1, format_func=label, key="stats_scatter_y")
    with c:
        color_by = st.selectbox("Colorear por", ["riesgo_oncologico", "sexo", "tabaquismo"], format_func=label, key="stats_scatter_color")
    complete = data.dropna(subset=list(dict.fromkeys([x, y]))).copy()
    sampled = complete.sample(n=min(2000, len(complete)), random_state=42).copy()
    sampled["grupo"] = sampled[color_by].map(lambda v: category(v, color_by))
    st.markdown("#### Relación entre dos variables")
    show_chart(sampled, {"mark": {"type": "point", "filled": True, "opacity": .45, "size": 23}, "encoding": {
        "x": {"field": x, "type": "quantitative", "title": label(x), "scale": {"zero": False}},
        "y": {"field": y, "type": "quantitative", "title": label(y), "scale": {"zero": False}},
        "color": {"field": "grupo", "type": "nominal", "title": label(color_by), "scale": {"range": COLORS}},
        "tooltip": [{"field": "id_paciente", "type": "nominal", "title": "Paciente sintético"}, {"field": x, "type": "quantitative", "title": label(x)}, {"field": y, "type": "quantitative", "title": label(y)}, {"field": "grupo", "type": "nominal"}]}}, height=350)
    st.caption(f"El diagrama muestra {len(sampled):,} de {len(complete):,} pares completos, mediante una muestra visual reproducible (semilla 42). Las correlaciones y el mapa de densidad usan todos los pares válidos de la selección.")
    st.markdown("#### Densidad de pacientes: todos los pares válidos")
    show_chart(complete[[x, y]].copy() if x != y else complete[[x]].copy(), {"mark": "rect", "encoding": {
        "x": {"field": x, "type": "quantitative", "bin": {"maxbins": 25}, "title": label(x)},
        "y": {"field": y, "type": "quantitative", "bin": {"maxbins": 25}, "title": label(y)},
        "color": {"aggregate": "count", "type": "quantitative", "title": "Pacientes", "scale": {"scheme": "blues"}},
        "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "Pacientes"}]}}, height=320)
    method = st.radio("Coeficiente de asociación numérica", ["spearman", "pearson"], format_func=lambda m: "Spearman · relación monótona" if m == "spearman" else "Pearson · relación lineal", horizontal=True, key="stats_correlation_method")
    correlations = stats.correlation_table(data, method=method)
    heat = correlations[["variable_x", "variable_y", "coeficiente", "n"]].copy()
    reverse = heat.rename(columns={"variable_x": "variable_y", "variable_y": "variable_x"})
    heat = pd.concat([heat, reverse], ignore_index=True)
    if not heat.empty:
        heat["variable_x"] = heat.variable_x.map(label)
        heat["variable_y"] = heat.variable_y.map(label)
    st.markdown("#### Matriz de correlaciones")
    show_chart(heat, {"mark": "rect", "encoding": {
        "x": {"field": "variable_x", "type": "nominal", "title": None, "axis": {"labelAngle": -40, "labelLimit": 200}},
        "y": {"field": "variable_y", "type": "nominal", "title": None, "axis": {"labelLimit": 200}},
        "color": {"field": "coeficiente", "type": "quantitative", "title": method.capitalize(), "scale": {"domain": [-1, 1], "scheme": "redblue", "reverse": True}},
        "tooltip": [{"field": "variable_x", "type": "nominal", "title": "Variable X"}, {"field": "variable_y", "type": "nominal", "title": "Variable Y"}, {"field": "coeficiente", "type": "quantitative", "format": ".3f"}, {"field": "n", "type": "quantitative", "title": "Pares válidos"}]}}, height=390)
    st.caption("Se excluyen pares sin suficiente variación o datos; la diagonal se omite. Una correlación describe asociación y no demuestra causalidad. Cada coeficiente usa sus propios pares completos.")
    with st.expander("Ver coeficientes, tamaños de muestra y valores p ajustados"):
        table(correlations)


def comparisons(data: pd.DataFrame):
    st.markdown("#### Proporción de etiqueta 1 por grupo")
    grouping = st.selectbox("Segmentación para estimaciones", GROUPS, format_func=label, key="stats_rates_group")
    rate_chart(data, grouping)
    if data.riesgo_oncologico.nunique() < 2:
        st.info("Las comparaciones entre etiquetas requieren que la selección contenga tanto la etiqueta 0 como la 1. Restablece el filtro de etiqueta para compararlas.")
        return
    st.markdown("#### Comparaciones numéricas: etiqueta 1 frente a etiqueta 0")
    numeric = stats.compare_numeric(data)
    table(numeric)
    st.caption("Welch compara medias sin exigir varianzas iguales; Mann–Whitney compara las distribuciones/rangos y no necesariamente las medianas. Diferencia e IC: media de etiqueta 1 menos media de etiqueta 0. Hedges g estandariza esa diferencia; el efecto biserial de rangos resume la separación ordinal. q aplica Benjamini–Hochberg por familia de pruebas.")
    st.markdown("#### Asociación entre categorías y etiqueta simulada")
    associations = stats.categorical_associations(data)
    table(associations)
    st.caption("Se usa chi-cuadrado, o Fisher cuando una tabla 2×2 tiene frecuencias esperadas pequeñas. V de Cramér mide intensidad de asociación, de 0 a 1. Revisa el estado de validez y las frecuencias esperadas antes de interpretar p o q.")
    st.info("Con 20.000 registros, diferencias pequeñas pueden producir valores p bajos. Interpreta conjuntamente tamaño del efecto, intervalo y magnitud; los resultados describen el mecanismo de simulación y no asociaciones clínicas verificadas.")
    contingency = pd.crosstab(data[grouping].map(lambda v: category(v, grouping)), data.riesgo_oncologico)
    within = contingency.div(contingency.sum(axis=1), axis=0).mul(100).rename_axis("grupo").reset_index().melt(id_vars="grupo", var_name="etiqueta", value_name="porcentaje")
    within["etiqueta"] = within.etiqueta.map(lambda v: category(v, "riesgo_oncologico"))
    st.markdown("#### Composición de la etiqueta dentro de cada grupo")
    show_chart(within, {"mark": "bar", "encoding": {
        "y": {"field": "grupo", "type": "nominal", "title": None},
        "x": {"field": "porcentaje", "type": "quantitative", "title": "Porcentaje dentro del grupo", "stack": "zero", "scale": {"domain": [0, 100]}},
        "color": {"field": "etiqueta", "type": "nominal", "title": "Etiqueta simulada", "scale": {"range": COLORS[:2]}},
        "tooltip": [{"field": "grupo", "type": "nominal"}, {"field": "etiqueta", "type": "nominal"}, {"field": "porcentaje", "type": "quantitative", "format": ".2f"}]}}, height=max(230, data[grouping].nunique()*40))
    a, b = st.columns(2)
    a.download_button("Descargar comparaciones numéricas", frame_csv(readable(numeric)), "comparaciones_numericas.csv", "text/csv", key="stats_download_numeric_tests")
    b.download_button("Descargar asociaciones categóricas", frame_csv(readable(associations)), "asociaciones_categoricas.csv", "text/csv", key="stats_download_category_tests")
    st.markdown("#### Odds Ratios e intervalos del 95%")
    odds = stats.odds_ratios(data)
    odds = odds.loc[odds.variable.eq(grouping)].copy()
    table(odds)
    finite = odds.loc[np.isfinite(odds["or"])].copy()
    if not finite.empty:
        finite["comparacion"] = [f"{category(v, grouping)} / {category(r, grouping)}" for v, r in zip(finite.categoria, finite.referencia)]
        show_chart(finite, {"layer": [
            {"mark": {"type": "rule", "color": COLORS[0]}, "encoding": {"y": {"field": "comparacion", "type": "nominal", "title": None}, "x": {"field": "ic_inf", "type": "quantitative", "scale": {"type": "log"}, "title": "Odds Ratio cruda · escala logarítmica"}, "x2": {"field": "ic_sup"}}},
            {"mark": {"type": "point", "filled": True, "color": COLORS[0], "size": 85}, "encoding": {"y": {"field": "comparacion", "type": "nominal", "title": None}, "x": {"field": "or", "type": "quantitative", "scale": {"type": "log"}}, "tooltip": [{"field": "comparacion", "title": "Comparación"}, {"field": "or", "type": "quantitative", "format": ".3f"}, {"field": "ic_inf", "type": "quantitative", "format": ".3f"}, {"field": "ic_sup", "type": "quantitative", "format": ".3f"}]}},
            {"mark": {"type": "rule", "color": "#697d8b", "strokeDash": [4, 4]}, "encoding": {"x": {"datum": 1, "type": "quantitative"}}}
        ]}, height=max(180, len(finite)*45))
    st.caption("OR no ajustadas: comparan los momios de etiqueta 1 de cada categoría con la referencia indicada. OR = ad/bc; IC aproximado en escala logarítmica. Se añade 0,5 a las cuatro celdas si hay ceros y se identifica la corrección. No son riesgos relativos, efectos causales ni coeficientes del modelo penalizado. Si falta la referencia en la selección, no se estima el contraste.")
    st.download_button("Descargar Odds Ratios", frame_csv(readable(odds)), "odds_ratios.csv", "text/csv", key="stats_download_odds")


def periods(data: pd.DataFrame):
    st.info("Cada paciente tiene un único registro. Los semestres representan grupos de procedencia: estas gráficas comparan cohortes de distintos periodos y no reconstruyen la evolución individual.")
    counts = data.groupby("periodo", observed=True).size().sort_index().rename("pacientes").reset_index()
    st.markdown("#### Volumen registrado por semestre")
    show_chart(counts, {"mark": {"type": "bar", "color": COLORS[0]}, "encoding": {
        "x": {"field": "periodo", "type": "ordinal", "sort": counts.periodo.tolist(), "title": "Semestre de procedencia", "axis": {"labelAngle": 0}},
        "y": {"field": "pacientes", "type": "quantitative", "title": "Pacientes", "scale": {"zero": True}},
        "tooltip": [{"field": "periodo", "type": "ordinal", "title": "Semestre"}, {"field": "pacientes", "type": "quantitative"}]}})
    st.markdown("#### Etiqueta 1 por semestre e incertidumbre")
    rate_chart(data, "periodo")
    variable = st.selectbox("Variable para comparar entre semestres", NUMERIC, format_func=label, key="stats_period_variable")
    period_summary = data.groupby("periodo", observed=True)[variable].agg(n="count", media="mean", mediana="median", desviacion="std").reset_index().sort_values("periodo")
    table(period_summary.rename(columns={"periodo": "Semestre"}).round(3))
    show_chart(data[["periodo", variable]], {"mark": {"type": "boxplot", "extent": 1.5, "outliers": False, "color": COLORS[0]}, "encoding": {
        "x": {"field": "periodo", "type": "ordinal", "sort": counts.periodo.tolist(), "title": "Semestre", "axis": {"labelAngle": 0}},
        "y": {"field": variable, "type": "quantitative", "title": label(variable), "scale": {"zero": False}}}})
    st.markdown("#### Media temporal, incertidumbre y media móvil de 2 semestres")
    timeline = stats.temporal_summary(data, variable)
    show_chart(timeline, {"layer": [
        {"mark": {"type": "area", "opacity": .16, "color": COLORS[0]}, "encoding": {
            "x": {"field": "periodo", "type": "ordinal", "sort": timeline.periodo.tolist(), "axis": {"labelAngle": 0}},
            "y": {"field": "ic_inf", "type": "quantitative", "title": label(variable), "scale": {"zero": False}}, "y2": {"field": "ic_sup"}, "detail": {"field": "tramo", "type": "nominal"}}},
        {"mark": {"type": "line", "point": True, "color": COLORS[0]}, "encoding": {
            "x": {"field": "periodo", "type": "ordinal", "sort": timeline.periodo.tolist()},
            "y": {"field": "media", "type": "quantitative", "title": label(variable)}, "detail": {"field": "tramo", "type": "nominal"},
            "tooltip": [{"field": "periodo"}, {"field": "n", "type": "quantitative"}, {"field": "media", "type": "quantitative", "format": ".3f"}, {"field": "ic_inf", "type": "quantitative", "format": ".3f"}, {"field": "ic_sup", "type": "quantitative", "format": ".3f"}]}},
        {"mark": {"type": "line", "point": True, "color": COLORS[1], "strokeDash": [6, 4]}, "encoding": {
            "x": {"field": "periodo", "type": "ordinal", "sort": timeline.periodo.tolist()},
            "y": {"field": "media_movil", "type": "quantitative", "title": label(variable)}, "detail": {"field": "tramo", "type": "nominal"},
            "tooltip": [{"field": "periodo"}, {"field": "media_movil", "type": "quantitative", "format": ".3f"}, {"field": "n_movil", "type": "quantitative"}, {"field": "movil_inf", "type": "quantitative", "format": ".3f"}, {"field": "movil_sup", "type": "quantitative", "format": ".3f"}]}}
    ]}, height=330)
    st.caption("Azul: media de cada cohorte y banda de IC t del 95%. Naranja discontinua: media móvil de los registros de dos semestres consecutivos, ponderada por sus tamaños. Sus IC se muestran en el tooltip y la tabla. El primer semestre y los huecos no tienen media móvil; no se interpolan pacientes ni se supone seguimiento individual. Solo hay cuatro semestres simulados: no permiten estimar estacionalidad ni pronósticos fiables.")
    with st.expander("Ver valores temporales e intervalos"):
        table(timeline.round(4))
    st.download_button("Descargar estimaciones temporales", frame_csv(timeline), "analisis_temporal.csv", "text/csv", key="stats_download_temporal")


@st.cache_data(show_spinner=False, max_entries=6)
def cached_model_diagnostics(_bundle, model_name: str, model_fingerprint: str, threshold: float):
    del model_name, model_fingerprint
    holdout = recover_holdout(_bundle, ANALYTICAL_DIR / "clinical_analytical.parquet")
    return diagnostic_results(holdout, threshold=threshold), holdout


def _model_diagnostics(bundle, threshold: float) -> dict:
    import hashlib
    import pickle
    fingerprint = hashlib.sha256(pickle.dumps(bundle) + (ANALYTICAL_DIR / "clinical_analytical.parquet").read_bytes()).hexdigest()
    return cached_model_diagnostics(bundle, bundle["model_name"], fingerprint, threshold)


def model_evaluation(bundle):
    st.markdown("#### Evaluación del modelo en la prueba reservada original")
    st.info("Esta vista utiliza la prueba reservada (20% de los pacientes de la base ANALYTICAL con la que se entrenó el modelo). Los filtros de la cohorte no modifican esta prueba. Se reconstruye la partición y se comprueba que reproduce las métricas guardadas; el modelo no se reentrena.")
    if bundle is None:
        st.warning("No hay un modelo persistido disponible.")
        return
    selected = float(bundle.get("threshold", .5))
    threshold = st.slider("Umbral de clasificación del modelo", .05, .95, selected, .01, key="stats_model_threshold",
                          help=f"Por defecto, {selected:.2f}: el umbral elegido con datos de desarrollo (máx. F1 fuera de pliegue).")
    import hashlib
    import pickle
    try:
        fingerprint = hashlib.sha256(pickle.dumps(bundle) + (ANALYTICAL_DIR / "clinical_analytical.parquet").read_bytes()).hexdigest()
    except OSError as exc:
        st.error(f"No se encuentra la base analítica de la prueba: {exc}")
        return
    try:
        with st.spinner("Comprobando la prueba reservada y sus intervalos…"):
            results, predictions = cached_model_diagnostics(bundle, bundle["model_name"], fingerprint, threshold)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"No se puede verificar la evaluación: {exc}")
        return
    metrics = results["metrics"]
    by_name = metrics.set_index("Métrica")["Estimación"]
    cards = st.columns(5)
    for container, name in zip(cards, ["ROC-AUC", "Accuracy (exactitud)", "Precision (VPP)", "Recall (sensibilidad)", "F1-Score"]):
        value = by_name[name]
        container.metric(name, f"{value:.3f}" if pd.notna(value) else "No definido")
    st.caption(f"Prueba: {results['n']} pacientes; etiqueta 1: {results['positive']}; umbral exploratorio: {threshold:.2f}. El umbral cambia la clasificación, no ROC-AUC. No se optimiza sobre esta prueba ni se presenta como recomendación clínica.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Matriz de confusión")
        matrix = results["matrix"]
        show_chart(matrix, {"layer": [
            {"mark": "rect", "encoding": {"x": {"field": "Clasificación del modelo", "type": "nominal", "axis": {"labelAngle": 0}}, "y": {"field": "Etiqueta observada", "type": "nominal"}, "color": {"field": "Pacientes", "type": "quantitative", "scale": {"scheme": "blues"}}, "tooltip": [{"field": "Etiqueta observada"}, {"field": "Clasificación del modelo"}, {"field": "Pacientes", "type": "quantitative"}]}},
            {"mark": {"type": "text", "fontSize": 20, "fontWeight": "bold"}, "encoding": {"x": {"field": "Clasificación del modelo", "type": "nominal"}, "y": {"field": "Etiqueta observada", "type": "nominal"}, "text": {"field": "Pacientes", "type": "quantitative"}, "color": {"condition": {"test": "datum.Pacientes > 70", "value": "white"}, "value": "#123958"}}}
        ]}, height=280)
    with right:
        st.markdown("#### Curva ROC")
        show_chart(results["roc"], {"layer": [
            {"mark": {"type": "line", "color": COLORS[0], "strokeWidth": 2}, "encoding": {"x": {"field": "Falsos positivos", "type": "quantitative", "title": "Tasa de falsos positivos", "scale": {"domain": [0, 1]}}, "y": {"field": "Sensibilidad", "type": "quantitative", "scale": {"domain": [0, 1]}}, "tooltip": [{"field": "Falsos positivos", "type": "quantitative", "title": "Tasa de falsos positivos"}, {"field": "Sensibilidad", "type": "quantitative"}]}},
            {"mark": {"type": "line", "color": "#83939d", "strokeDash": [4, 4]}, "encoding": {"x": {"field": "Falsos positivos", "type": "quantitative"}, "y": {"field": "Falsos positivos", "type": "quantitative"}}}
        ]}, height=280)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Curva Precision–Recall")
        show_chart(results["pr"], {"mark": {"type": "line", "color": COLORS[0]}, "encoding": {"x": {"field": "Recall", "type": "quantitative", "scale": {"domain": [0, 1]}}, "y": {"field": "Precision", "type": "quantitative", "scale": {"domain": [0, 1]}}, "tooltip": [{"field": "Recall", "type": "quantitative"}, {"field": "Precision", "type": "quantitative"}]}})
    with right:
        st.markdown("#### Calibración de probabilidades")
        show_chart(results["calibration"], {"layer": [
            {"mark": {"type": "line", "point": True, "color": COLORS[0]}, "encoding": {"x": {"field": "Probabilidad estimada", "type": "quantitative", "scale": {"domain": [0, 1]}}, "y": {"field": "Proporción observada", "type": "quantitative", "scale": {"domain": [0, 1]}}, "tooltip": [{"field": "Probabilidad estimada", "type": "quantitative"}, {"field": "Proporción observada", "type": "quantitative"}]}},
            {"mark": {"type": "line", "color": "#83939d", "strokeDash": [4, 4]}, "encoding": {"x": {"field": "Probabilidad estimada", "type": "quantitative"}, "y": {"field": "Probabilidad estimada", "type": "quantitative"}}}
        ]})
    table(metrics)
    st.caption("IC del 95% por bootstrap percentil estratificado, 500 réplicas, semilla 42; condicionado a este modelo fijo y a la proporción de clases observada. No incluye incertidumbre de entrenamiento ni validación externa. Valores sin predicciones positivas se conservan como no definidos. La calibración usa 8 grupos de probabilidad; Brier menor es mejor.")
    st.download_button("Descargar métricas e intervalos del modelo", frame_csv(metrics), "evaluacion_prueba_reservada.csv", "text/csv", key="stats_download_model")


@st.cache_data(show_spinner=False, max_entries=6)
def multivariate_result(data: pd.DataFrame, k: int):
    return stats.pca_analysis(data), stats.clustering_analysis(data, k=k)


def multivariate(data: pd.DataFrame):
    st.caption("Estas técnicas usan únicamente las seis variables numéricas, estandarizadas a media 0 y desviación 1. La etiqueta simulada no se utiliza para construir los componentes ni los grupos.")
    k = st.slider("Número de grupos exploratorios (K-means)", min_value=2, max_value=6, value=3, key="stats_cluster_k")
    with st.spinner("Calculando componentes y segmentación de la selección…"):
        pca, clusters = multivariate_result(data, k)
    st.markdown("#### Análisis de componentes principales (PCA)")
    variance = pca["variance"]
    st.caption(f"Registros completos utilizados: {pca['n']:,}; excluidos por datos incompletos/no finitos: {pca['excluded']:,}.")
    if variance.empty:
        st.info(str(pca.get("status", "Datos insuficientes para PCA.")))
    else:
        show_chart(variance, {"layer": [
            {"mark": {"type": "bar", "color": COLORS[0]}, "encoding": {"x": {"field": "componente", "type": "ordinal", "sort": None, "title": "Componente"}, "y": {"field": "varianza_pct", "type": "quantitative", "title": "Varianza explicada (%)", "scale": {"domain": [0, 100]}}, "tooltip": [{"field": "componente", "type": "ordinal"}, {"field": "varianza_pct", "type": "quantitative", "format": ".2f"}]}},
            {"mark": {"type": "line", "point": True, "color": COLORS[1]}, "encoding": {"x": {"field": "componente", "type": "ordinal", "sort": None}, "y": {"field": "acumulada_pct", "type": "quantitative", "title": "Varianza explicada (%)"}, "tooltip": [{"field": "acumulada_pct", "type": "quantitative", "title": "Acumulada (%)", "format": ".2f"}]}}
        ]})
        scores = pca["scores"]
        if "PC2" in scores:
            display = scores.sample(n=min(2000, len(scores)), random_state=42).copy()
            display["etiqueta"] = display.riesgo_oncologico.map(lambda v: category(v, "riesgo_oncologico"))
            show_chart(display, {"mark": {"type": "point", "filled": True, "opacity": .45, "size": 24}, "encoding": {
                "x": {"field": "PC1", "type": "quantitative", "title": "Componente principal 1"},
                "y": {"field": "PC2", "type": "quantitative", "title": "Componente principal 2"},
                "color": {"field": "etiqueta", "type": "nominal", "title": "Etiqueta simulada", "scale": {"range": COLORS[:2]}},
                "tooltip": [{"field": "id_paciente", "type": "nominal", "title": "Paciente sintético"}, {"field": "PC1", "type": "quantitative"}, {"field": "PC2", "type": "quantitative"}]}})
            st.caption(f"Proyección visual: {len(display):,} puntos muestreados; el ajuste PCA usa los {pca['n']:,} registros completos. Dos ejes pueden representar solo una parte de la variación total.")
        with st.expander("Ver pesos de las variables y varianza explicada"):
            table(readable(pca["loadings"]).round(4))
            table(variance.round(3))
        st.caption("Los pesos indican cómo se construye cada componente; no son efectos causales ni medidas de riesgo.")
    st.markdown("#### Segmentación exploratoria con K-means")
    if clusters["summary"].empty:
        st.info(str(clusters.get("status", "Datos insuficientes para la segmentación.")))
    else:
        cluster_summary = clusters["summary"].copy()
        cluster_summary["grupo"] = "Grupo " + cluster_summary.cluster.astype(str)
        show_chart(cluster_summary, {"mark": {"type": "bar", "color": COLORS[0]}, "encoding": {
            "x": {"field": "grupo", "type": "nominal", "title": "Grupo exploratorio", "axis": {"labelAngle": 0}},
            "y": {"field": "n", "type": "quantitative", "title": "Pacientes"},
            "tooltip": [{"field": "grupo", "type": "nominal"}, {"field": "n", "type": "quantitative"}, {"field": "etiqueta1_pct", "type": "quantitative", "title": "Etiqueta 1 (%)", "format": ".2f"}]}})
        table(cluster_summary.drop(columns="cluster").round(3))
        st.markdown("**Centros de los grupos en las unidades originales**")
        table(clusters["centroids"].rename(columns=LABELS).round(3))
        silhouette = clusters.get("silhouette", np.nan)
        if pd.notna(silhouette):
            st.caption(f"Silueta aproximada: {silhouette:.3f} (muestra reproducible de hasta 1.000 registros; rango −1 a 1). Compara separación y cohesión, no utilidad clínica. K-means usa {clusters['n']:,} registros completos; excluye {clusters['excluded']:,}.")
        st.caption("Los números de grupo son identificadores arbitrarios. No son categorías de riesgo ni diagnósticos. Cambiar filtros o k produce una segmentación nueva.")


def quality_and_data(data: pd.DataFrame, context: dict, metadata: dict):
    report = stats.quality_report(data)
    columns = report["columns"]
    summary = report["summary"]
    a, b, c, d = st.columns(4)
    a.metric("Celdas ausentes", str(summary["missing_cells"]))
    b.metric("IDs repetidos adicionales", str(summary["duplicate_ids"]))
    c.metric("Celdas fuera de dominio", str(summary["invalid_cells"]))
    d.metric("Sexo M y paridad > 0", str(summary["contradictions"]))
    st.markdown("#### Calidad por variable")
    missing = readable(columns).copy()
    if summary["missing_cells"] == 0:
        st.success("No hay celdas ausentes en las variables de esta selección.")
    else:
        show_chart(missing, {"mark": {"type": "bar", "color": COLORS[0]}, "encoding": {
            "y": {"field": "variable", "type": "nominal", "title": None, "axis": {"labelLimit": 220}},
            "x": {"field": "ausentes_pct", "type": "quantitative", "title": "Valores ausentes (%)", "scale": {"domain": [0, 100]}},
            "tooltip": [{"field": "variable", "type": "nominal"}, {"field": "ausentes", "type": "quantitative"}, {"field": "ausentes_pct", "type": "quantitative", "format": ".2f"}]}}, height=max(300, len(missing)*27))
    table(missing.round(3))
    outliers = readable(columns.loc[columns.variable.isin(NUMERIC)]).copy()
    st.markdown("#### Observaciones atípicas por variable")
    show_chart(outliers, {"mark": {"type": "bar", "color": COLORS[1]}, "encoding": {
        "y": {"field": "variable", "type": "nominal", "title": None},
        "x": {"field": "atipicos_iqr", "type": "quantitative", "title": "Pacientes fuera de los límites IQR"},
        "tooltip": [{"field": "variable", "type": "nominal"}, {"field": "atipicos_iqr", "type": "quantitative"}]}})
    st.caption("Atípicos: valores fuera de Q1−1,5×IQR o Q3+1,5×IQR en cada variable numérica. Son señales de revisión, no errores automáticos; no se eliminan ni se imputan registros para este panel.")
    st.markdown("#### Controles de consistencia y validez")
    table(report["issues"].round(3))
    if summary["contradictions"]:
        st.warning(f"Hay {summary['contradictions']:,} registros con sexo M y paridad mayor que 0, combinación generada de manera independiente en la base sintética. Se conservan y se señalan para revisar la coherencia de esas variables; los análisis de paridad requieren esta salvedad.")
    st.markdown("#### Inspección de pacientes sintéticos")
    selected_id = st.text_input("Buscar identificador exacto de paciente", placeholder="Ejemplo: PAC-00001", key="stats_patient_id").strip()
    if selected_id:
        patient = data.loc[data.id_paciente.astype(str).str.casefold().eq(selected_id.casefold())]
        if patient.empty:
            st.info("Ese identificador no está presente en la selección actual.")
        else:
            table(patient.rename(columns=LABELS))
            st.caption("Este registro contiene una observación. No hay consultas repetidas fechadas para construir un historial individual.")
    with st.expander("Ver registros de la selección"):
        st.caption(f"Vista previa de hasta 500 registros de los {len(data):,} seleccionados. La descarga incluye toda la selección.")
        table(data.head(500).rename(columns=LABELS))
    a, b = st.columns(2)
    with a:
        st.download_button("Descargar pacientes de la selección", frame_csv(data), "pacientes_seleccion.csv", "text/csv", key="stats_download_patients")
    with b:
        evidence = {"clasificacion": "Datos sintéticos", "fuente": metadata.get("file_name", "pacientes_sinteticos_20000.csv"), "sha256": metadata.get("sha256", metadata.get("hash", "")), "filtros": context, "nota": "Una fila por paciente ficticio; etiqueta simulada sin horizonte clínico. Estadísticos sin imputación ni eliminación de atípicos."}
        st.download_button("Descargar filtros y procedencia", json.dumps(evidence, ensure_ascii=False, indent=2, default=str).encode("utf-8"), "contexto_analisis.json", "application/json", key="stats_download_context")
    with st.expander("Métodos, fórmulas y límites del análisis"):
        st.markdown(r"""
        - **Unidad de análisis:** una fila por paciente ficticio. Los filtros se intersectan y delimitan toda la población analizada y exportada.
        - **Frecuencias:** conteos y porcentajes sobre el total seleccionado; en tasas por grupo, el denominador es el número de etiquetas 0/1 válidas de ese grupo.
        - **Descriptivos:** media, mediana, varianza y desviación muestrales, cuartiles, IQR, extremos, coeficiente de variación, asimetría y curtosis en exceso. Los ausentes se contabilizan y se excluyen solo del cálculo que necesita ese dato.
        - **Intervalo de la media:** media ± t(0,975; n−1) × desviación/√n. Requiere independencia y una aproximación de muestreo; con datos escasos o constantes se informa el estado.
        - **Proporciones:** intervalo de Wilson del 95%, válido también ante proporciones 0 o 1. No es un intervalo de predicción individual.
        - **Comparaciones:** Welch para diferencia de medias, Mann–Whitney para rangos/distribuciones; Hedges g y efecto biserial de rangos para magnitud. IC de Welch para media1−media0.
        - **Normalidad:** Shapiro–Wilk sobre hasta 5.000 valores válidos por variable, muestra con semilla 42 y ajuste BH entre las seis pruebas. La KDE usa núcleos gaussianos y ancho de banda de Scott; es una representación suavizada, no una hipótesis normal.
        - **Odds Ratios:** OR crudas con referencia explícita, OR=ad/bc e IC log-Wald del 95%. Ante celdas nulas se añade 0,5 a toda la tabla. No se interpretan como efectos ajustados ni como riesgo relativo. q de las OR se ajusta en conjunto para todos los contrastes categóricos calculados, aunque se muestre una variable a la vez.
        - **Semestres:** medias e IC t por periodo; ventana móvil de dos semestres consecutivos agrupando los registros originales, con ponderación por tamaño. Los huecos no se interpolan y el primer periodo no tiene ventana completa.
        - **Evaluación del modelo:** recupera la prueba reservada original y coteja AUROC y matriz de confusión con las métricas persistidas. IC por bootstrap percentil estratificado de 500 réplicas, condicionado al modelo y proporción de clases de esa prueba; no incluye variación del entrenamiento ni equivale a validación externa.
        - **Asociaciones:** Pearson para relaciones lineales; Spearman para relaciones monótonas. Chi-cuadrado/Fisher para categorías y V de Cramér para intensidad. q ajusta por Benjamini–Hochberg dentro de cada familia visible, no entre todas las exploraciones posibles.
        - **Multivariante:** PCA y K-means sobre casos completos con seis variables estandarizadas, semilla 42 y sin utilizar la etiqueta para ajustar. Las cifras de grupo pueden cambiar al filtrar.
        - **Alcance:** los intervalos y contrastes ilustran métodos bajo sus supuestos. Como los datos se simularon, no establecen prevalencia hospitalaria, causalidad, eficacia ni riesgo clínico. No hay seguimiento individual, desenlace fechado, tratamientos ni censura: supervivencia, Kaplan–Meier, Cox y evolución clínica individual no se pueden estimar con esta estructura.
        """)


def _pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}".replace(".", ",") + " %"


def dashboard_summary(data: pd.DataFrame, bundle) -> None:
    """Resumen de una sola pantalla: cifras clave, tendencia, modelo y desglose."""
    valid = data.riesgo_oncologico.isin([0, 1])
    positives = data.loc[valid, "riesgo_oncologico"]
    prevalence = float(positives.mean()) if len(positives) else float("nan")
    suspicious = float((data.resultado_imagen == "sospechoso").mean())
    family = float((data.antecedente_familiar_cancer == 1).mean())
    kpis = [
        ("Pacientes", f"{data.id_paciente.nunique():,}".replace(",", "."), "en la selección actual"),
        ("Etiqueta 1", _pct(prevalence), f"{int(positives.sum()):,}".replace(",", ".") + " pacientes"),
        ("Edad promedio", f"{data.edad.mean():.1f}".replace(".", ",") + " años", f"mediana {data.edad.median():.0f}"),
        ("Imagen sospechosa", _pct(suspicious), f"antecedente familiar {_pct(family, 0)}"),
    ]
    cells = "".join(f'<div class="or-kpi"><span>{k}</span><b>{v}</b><small>{s}</small></div>' for k, v, s in kpis)
    st.html(f'<div class="or-kpis">{cells}</div>')

    left, right = st.columns([2.1, 1], gap="large")
    with left:
        st.html('<div class="or-dash-h"><h3>Etiqueta por semestre</h3><p>Pacientes con etiqueta 1 en cada semestre de procedencia.</p></div>')
        mode = st.segmented_control("Tipo de gráfico", ["Puntos", "Línea"], default="Puntos", key="stats_trend_mode", label_visibility="collapsed")
        trend = (data.loc[valid].groupby("periodo")["riesgo_oncologico"].agg(positivos="sum", pacientes="size").reset_index())
        trend["prevalencia"] = trend.positivos / trend.pacientes
        if mode == "Línea":
            show_chart(trend, {"layer": [
                {"mark": {"type": "area", "color": COLORS[0], "opacity": .12, "interpolate": "monotone"}},
                {"mark": {"type": "line", "color": COLORS[0], "strokeWidth": 2.5, "interpolate": "monotone", "point": {"filled": True, "size": 60, "color": COLORS[0]}}}],
                "encoding": {"x": {"field": "periodo", "type": "ordinal", "axis": {"labelAngle": 0}},
                             "y": {"field": "prevalencia", "type": "quantitative", "axis": {"format": ".0%"}, "scale": {"zero": False}},
                             "tooltip": [{"field": "periodo"}, {"field": "prevalencia", "type": "quantitative", "format": ".2%"}, {"field": "positivos", "type": "quantitative"}]}}, height=260)
        else:
            unit = max(1, int(np.ceil(trend.positivos.max() / 24))) if len(trend) else 1
            dots = pd.DataFrame([{"periodo": r.periodo, "nivel": i, "positivos": int(r.positivos)}
                                 for r in trend.itertuples() for i in range(int(round(r.positivos / unit)))])
            show_chart(dots, {"mark": {"type": "square", "size": 150, "color": COLORS[1], "opacity": .9},
                              "encoding": {"x": {"field": "periodo", "type": "ordinal", "axis": {"labelAngle": 0}, "scale": {"paddingInner": .55}},
                                           "y": {"field": "nivel", "type": "ordinal", "sort": "descending", "axis": None, "title": None},
                                           "tooltip": [{"field": "periodo"}, {"field": "positivos", "type": "quantitative", "title": "Pacientes con etiqueta 1"}]}}, height=260)
            st.caption(f"Cada cuadro representa unos {unit} pacientes con etiqueta 1.")
    with right:
        st.html('<div class="or-dash-h"><h3>Modelo seleccionado</h3><p>Discriminación en la prueba reservada.</p></div>')
        if bundle is None:
            st.caption("El modelo aún no se ha generado.")
        else:
            ev, null = bundle["evaluation"]["modelo"], bundle["evaluation"]["modelo_nulo"]
            auroc = float(ev["auroc"])
            gauge = pd.DataFrame({"parte": ["AUROC", "resto"], "valor": [auroc, 1 - auroc]})
            st.vega_lite_chart(gauge, {"height": 210, "config": CONFIG, "layer": [
                {"mark": {"type": "arc", "innerRadius": 72, "outerRadius": 92, "cornerRadius": 0},
                 "encoding": {"theta": {"field": "valor", "type": "quantitative", "stack": True},
                              "color": {"field": "parte", "type": "nominal", "scale": {"domain": ["AUROC", "resto"], "range": [COLORS[0], "#e4e1da"]}, "legend": None},
                              "order": {"field": "parte", "sort": "ascending"}}},
                {"mark": {"type": "text", "fontSize": 38, "fontWeight": 800, "font": "Archivo", "color": "#0d0f14"},
                 "data": {"values": [{"t": f"{auroc:.3f}".replace(".", ",")}]}, "encoding": {"text": {"field": "t"}}}]},
                width="stretch")
            dec = lambda v, n: f"{float(v):.{n}f}".replace(".", ",")
            st.html(f'''<dl class="or-mini-dl"><dt>Brier</dt><dd>{dec(ev["brier_score"], 3)} <small>nulo {dec(null["brier_score"], 3)}</small></dd>
                <dt>Sensibilidad</dt><dd>{_pct(ev["sensibilidad"], 0)}</dd><dt>Especificidad</dt><dd>{_pct(ev["especificidad"], 0)}</dd>
                <dt>Umbral</dt><dd>{dec(bundle.get("threshold", .5), 2)}</dd></dl>''')

    left, right = st.columns([2.1, 1], gap="large")
    with left:
        order = [("sospechoso", "Sospechoso", COLORS[1]), ("indeterminado", "Indeterminado", "#e3a008"), ("normal", "Normal", COLORS[2])]
        counts = data.resultado_imagen.value_counts()
        total = int(counts.sum()) or 1
        rates = data.loc[valid].groupby("resultado_imagen")["riesgo_oncologico"].mean()
        bar = "".join(f'<i style="width:{counts.get(k, 0) / total * 100:.2f}%;background:{c}"></i>' for k, _, c in order)
        miles = lambda v: f"{int(v):,}".replace(",", ".")
        rows = "".join(f'<li><span class="sw" style="background:{c}"></span>{name}<b>{_pct(counts.get(k, 0) / total)}</b>'
                       f'<em>{miles(counts.get(k, 0))} pacientes · etiqueta 1 en {_pct(rates.get(k, float("nan")))}</em></li>'
                       for k, name, c in order)
        st.html(f'<div class="or-dash-h"><h3>Hallazgo de imagen</h3><p>Composición de la selección y proporción con etiqueta 1 en cada grupo.</p></div>'
                f'<div class="or-stack">{bar}</div><ul class="or-legend">{rows}</ul>')
    with right:
        st.html('<div class="or-dash-h"><h3>Etiqueta 1 por edad</h3><p>Proporción en cada grupo de edad.</p></div>')
        by_age = data.loc[valid].groupby("grupo_etario", observed=True)["riesgo_oncologico"].mean().rename("prevalencia").reset_index()
        show_chart(by_age, {"mark": {"type": "bar", "color": COLORS[0], "width": {"band": .55}},
                            "encoding": {"x": {"field": "grupo_etario", "type": "ordinal", "sort": ["<40", "40-54", "55-69", "70+"], "axis": {"labelAngle": 0}},
                                         "y": {"field": "prevalencia", "type": "quantitative", "axis": {"format": ".0%"}},
                                         "tooltip": [{"field": "grupo_etario"}, {"field": "prevalencia", "type": "quantitative", "format": ".1%"}]}}, height=190)


def render_statistics_panel(bundle=None):
    """Renderiza el panel completo sin modificar la inferencia individual existente."""
    try:
        source, metadata = load_patient_data()
    except (OSError, ValueError) as exc:
        source, metadata, load_error = pd.DataFrame(), {}, exc
    else:
        load_error = None
    periods_text = " a ".join([min(metadata.get("periods") or ["—"]), max(metadata.get("periods") or ["—"])])
    checksum = str(metadata.get("sha256", metadata.get("hash", "")))
    st.html(f'''<div class="dk-top"><div><span class="dk-eyebrow">Análisis de la cohorte</span>
      <h1>Análisis estadístico e inferencial</h1>
      <p>{f"{len(source):,}".replace(",", ".")} pacientes ficticios · semestres {periods_text}</p></div>
      <div class="dk-meta"><span><b>Fuente</b>{metadata.get("file_name", "Sin fuente")}</span><span><b>Huella SHA-256</b><code>{checksum[:16]}…</code></span></div></div>''')
    with st.container(key="statistics-section"):
        if load_error is not None:
            st.error(f"No se pudo cargar la base del panel: {load_error}")
            return
        if source.empty:
            st.info("La fuente está vacía. No hay datos que analizar.")
            return
        source = source.copy()
        if "grupo_etario" not in source:
            source["grupo_etario"] = pd.cut(source.edad, [0, 40, 55, 70, 121], right=False, labels=["<40", "40–54", "55–69", "70+"])
        if "imc_categoria" not in source:
            source["imc_categoria"] = pd.cut(source.imc, [0, 18.5, 25, 30, 101], right=False, labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"])
        data, context = apply_filters(source)
        if data.empty:
            st.info("No hay pacientes que cumplan todos los filtros. Selecciona al menos una opción en cada filtro o pulsa Restablecer filtros.")
            st.download_button("Descargar selección vacía", frame_csv(data), "pacientes_seleccion.csv", "text/csv", key="stats_download_empty")
            return
        with st.container(key="stats_dashboard"):
            render_dashboard(data, bundle)
        st.html('<div class="dk-sep"><h2>Análisis detallado</h2><p>Cinco vistas: los datos, sus distribuciones, los factores asociados, el modelo y la calidad.</p></div>')
        selected_view = st.segmented_control("Vista de análisis", VIEWS, default=VIEWS[0], key="stats_active_view",
                                             selection_mode="single", label_visibility="collapsed", bind="query-params")
        with st.container(key="stats_detail"):
            if selected_view == "Cohorte":
                view_cohort(data)
            elif selected_view == "Distribuciones":
                view_distributions(data)
            elif selected_view == "Factores asociados":
                view_factors(data)
            elif selected_view == "Modelo":
                view_model(bundle, _model_diagnostics, source)
            elif selected_view == "Calidad de datos":
                view_quality(data, stats.quality_report(data), context, metadata)
