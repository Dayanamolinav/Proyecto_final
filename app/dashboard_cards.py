"""Resumen visual del panel de análisis: tarjetas de cifras, gráficas ECharts e infografía.

Todo se calcula sobre la selección filtrada; las gráficas usan st.echarts_chart con
los datos en dataset.source, como recomienda la documentación de Streamlit.
"""
from __future__ import annotations

import base64
import html

import numpy as np
import pandas as pd
import streamlit as st

GREEN, GREEN2, LIME, CORAL, INK, MUTE = "#14946a", "#1fb985", "#b5dd4a", "#ff8a5c", "#16231d", "#7b8a83"
AGE_ORDER = ["<40", "40–54", "55–69", "70+"]
AXIS = {"axisLine": {"show": False}, "axisTick": {"show": False},
        "axisLabel": {"color": "#9aa8a1", "fontSize": 11}, "splitLine": {"lineStyle": {"color": "#edf2ef"}}}


def _svg_img(svg: str, cls: str, alt: str = "") -> str:
    """st.html limpia los <svg> en línea; se insertan como imagen con URI de datos."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="{cls}" alt="{alt}" src="data:image/svg+xml;base64,{data}">'


def _miles(v) -> str:
    return f"{int(v):,}".replace(",", ".")


def _pct(v: float, d: int = 1) -> str:
    return "Sin dato" if v is None or np.isnan(v) else f"{v * 100:.{d}f}".replace(".", ",") + " %"


def _ticks(share: float, color: str, light: str, n: int = 36) -> str:
    """Tira de rayitas: la parte coloreada representa la proporción indicada."""
    on = int(round(max(0.0, min(1.0, share)) * n))
    heights = [55 + int(45 * abs(np.sin(i * 1.7))) for i in range(n)]
    return "".join(f'<i style="height:{h}%;background:{color if i < on else light}"></i>' for i, h in enumerate(heights))


def kpi_cards(data: pd.DataFrame) -> None:
    valid = data.riesgo_oncologico.isin([0, 1])
    pos = int(data.loc[valid, "riesgo_oncologico"].sum())
    n_valid = int(valid.sum()) or 1
    n = data.id_paciente.nunique()
    women = int((data.sexo == "F").sum())
    suspicious = int((data.resultado_imagen == "sospechoso").sum())
    ages = data.edad.dropna()
    cards = [
        ("Pacientes", _miles(n), _ticks(women / max(len(data), 1), GREEN2, "#c9eadb"),
         f"Mujeres <b>{_miles(women)}</b>", f"Hombres <b>{_miles(len(data) - women)}</b>"),
        ("Etiqueta 1", _pct(pos / n_valid), _ticks(pos / n_valid, CORAL, "#f6ddd2"),
         f"Positivos <b>{_miles(pos)}</b>", f"Negativos <b>{_miles(n_valid - pos)}</b>"),
        ("Edad promedio", f"{ages.mean():.1f}".replace(".", ",") + " años", _ticks((ages.mean() - 18) / 82, LIME, "#e8f3c8"),
         f"Mínima <b>{ages.min():.0f}</b>", f"Máxima <b>{ages.max():.0f}</b>"),
        ("Imagen sospechosa", _pct(suspicious / max(len(data), 1)), _ticks(suspicious / max(len(data), 1), GREEN, "#cfe6dc"),
         f"Sospechosa <b>{_miles(suspicious)}</b>", f"Otras <b>{_miles(len(data) - suspicious)}</b>"),
    ]
    st.html('<div class="dk-kpis">' + "".join(
        f'<div class="dk-card dk-kpi"><small>{t}</small><b>{v}</b><div class="dk-ticks">{tk}</div>'
        f'<div class="dk-tl"><span>{a}</span><span>{b}</span></div></div>' for t, v, tk, a, b in cards) + "</div>")


def trend_chart(data: pd.DataFrame) -> None:
    valid = data.riesgo_oncologico.isin([0, 1])
    trend = (data.loc[valid].groupby("periodo")["riesgo_oncologico"].agg(positivos="sum", pacientes="size").reset_index())
    trend["prevalencia"] = (trend.positivos / trend.pacientes * 100).round(1)
    trend["periodo"] = trend.periodo.astype(str)
    last = trend.iloc[-1] if len(trend) else None
    delta = (trend.prevalencia.iloc[-1] - trend.prevalencia.iloc[-2]) if len(trend) > 1 else 0.0
    arrow, color = ("↑", CORAL) if delta > 0 else ("↓", GREEN)
    head = (f'<div class="dk-big"><b>{last.prevalencia:.1f} %</b>'.replace(".", ",") +
            f'<span style="color:{color}">{arrow} {abs(delta):.1f} pts frente al semestre anterior</span></div>'.replace(".", ",")
            if last is not None else "")
    st.html(f'<div class="dk-h"><h3>Etiqueta 1 por semestre</h3><p>Proporción de pacientes con etiqueta 1 en cada semestre</p></div>{head}')
    st.echarts_chart({
        "grid": {"left": 8, "right": 16, "top": 16, "bottom": 8, "containLabel": True},
        "tooltip": {"trigger": "axis"},
        "dataset": {"source": trend[["periodo", "prevalencia"]]},
        "xAxis": {"type": "category", "boundaryGap": False, **AXIS, "splitLine": {"show": False}},
        "yAxis": {"type": "value", "scale": True, **AXIS, "axisLabel": {"color": "#9aa8a1", "fontSize": 11, "formatter": "{value} %"}},
        "series": [{"type": "line", "smooth": True, "symbol": "circle", "symbolSize": 8, "showSymbol": False,
                    "lineStyle": {"width": 3, "color": GREEN}, "itemStyle": {"color": GREEN, "borderColor": "#fff", "borderWidth": 2},
                    "emphasis": {"focus": "series"},
                    "areaStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                                            "colorStops": [{"offset": 0, "color": "rgba(31,185,133,.35)"}, {"offset": 1, "color": "rgba(31,185,133,0)"}]}},
                    "markPoint": {"symbol": "pin", "symbolSize": 44, "data": [{"type": "max", "name": "Máximo"}],
                                  "itemStyle": {"color": INK}, "label": {"fontSize": 10, "color": "#fff", "formatter": "{c}"}}}],
    }, height=250, alt="Proporción de etiqueta 1 por semestre de procedencia")


def model_card(bundle) -> None:
    st.html('<div class="dk-h"><h3>Modelo seleccionado</h3><p>Discriminación en la prueba reservada</p></div>')
    if bundle is None:
        st.caption("El modelo aún no se ha generado.")
        return
    ev, null = bundle["evaluation"]["modelo"], bundle["evaluation"]["modelo_nulo"]
    rings = pd.DataFrame({"medida": ["AUROC", "Sensibilidad"], "valor": [round(float(ev["auroc"]) * 100, 1), round(float(ev["sensibilidad"]) * 100, 1)]})

    def ring(center: str, value: float, top: float, fmt: str, name: str, color: str, track: str) -> dict:
        return {"type": "gauge", "min": 0, "max": top, "center": [center, "44%"], "radius": "66%", "startAngle": 90, "endAngle": -270,
                "pointer": {"show": False}, "progress": {"show": True, "roundCap": True, "width": 12, "itemStyle": {"color": color}},
                "axisLine": {"lineStyle": {"width": 12, "color": [[1, track]]}}, "splitLine": {"show": False},
                "axisTick": {"show": False}, "axisLabel": {"show": False},
                "title": {"offsetCenter": [0, "128%"], "fontSize": 12, "color": MUTE},
                "detail": {"offsetCenter": [0, 0], "fontSize": 22, "fontWeight": 800, "color": INK, "formatter": fmt},
                "data": [{"value": value, "name": name}]}

    st.echarts_chart({"series": [ring("27%", round(float(ev["auroc"]), 3), 1, "{value}", "AUROC", GREEN, "#e8f1ec"),
                                 ring("73%", float(rings.valor[1]), 100, "{value} %", "Sensibilidad", LIME, "#f1f7df")]},
                     height=200, alt="Anillos con el AUROC y la sensibilidad del modelo")
    dec = lambda v, n: f"{float(v):.{n}f}".replace(".", ",")
    st.html(f'''<div class="dk-rows"><div><span>AUROC</span><b>{dec(ev["auroc"], 3)}</b></div>
      <div><span>Brier</span><b>{dec(ev["brier_score"], 3)} <em>· nulo {dec(null["brier_score"], 3)}</em></b></div>
      <div><span>Especificidad</span><b>{_pct(ev["especificidad"], 0)}</b></div>
      <div><span>Umbral</span><b>{dec(bundle.get("threshold", .5), 2)}</b></div></div>''')


def age_chart(data: pd.DataFrame) -> None:
    valid = data.riesgo_oncologico.isin([0, 1])
    by = (data.loc[valid].groupby("grupo_etario", observed=True)["riesgo_oncologico"].agg(tasa="mean", pacientes="size").reset_index())
    by["grupo_etario"] = by.grupo_etario.astype(str).str.replace("-", "–")
    by = by.set_index("grupo_etario").reindex([a for a in AGE_ORDER if a in set(by.grupo_etario)]).reset_index()
    by["tasa"] = (by.tasa * 100).round(1)
    top = by.tasa.idxmax() if len(by) else None
    st.html('<div class="dk-h"><h3>Etiqueta 1 por edad</h3><p>Proporción en cada grupo de edad</p></div>')
    stripes = {"decal": {"symbol": "rect", "dashArrayX": [1, 0], "dashArrayY": [4, 3], "rotation": -0.785, "color": "rgba(255,255,255,.35)"}}
    st.echarts_chart({
        "grid": {"left": 4, "right": 4, "top": 28, "bottom": 4, "containLabel": True},
        "tooltip": {"trigger": "axis"},
        "dataset": {"source": by[["grupo_etario", "tasa", "pacientes"]]},
        "xAxis": {"type": "category", **AXIS, "splitLine": {"show": False}},
        "yAxis": {"type": "value", "show": False},
        "series": [{"type": "bar", "encode": {"x": "grupo_etario", "y": "tasa"}, "barWidth": "58%",
                    "itemStyle": {"color": GREEN2, "borderRadius": [8, 8, 0, 0], **stripes},
                    "label": {"show": True, "position": "top", "color": INK, "fontWeight": 700, "formatter": "{@tasa} %"},
                    }],
        "visualMap": {"show": False, "dimension": 1, "pieces": [{"gte": float(by.tasa.max()) if top is not None else 0, "color": LIME}], "outOfRange": {"color": GREEN2}},
    }, height=230, alt="Proporción de etiqueta 1 por grupo de edad")


def image_card(data: pd.DataFrame) -> None:
    valid = data.riesgo_oncologico.isin([0, 1])
    counts = data.resultado_imagen.value_counts()
    rates = data.loc[valid].groupby("resultado_imagen")["riesgo_oncologico"].mean()
    rows = [("sospechoso", "Sospechoso", "S", CORAL), ("indeterminado", "Indeterminado", "I", LIME), ("normal", "Normal", "N", GREEN2)]
    items = "".join(
        f'<li><span class="dk-ic">{letter}</span><div><small>{name} · {_miles(counts.get(k, 0))} pacientes</small>'
        f'<div class="dk-track"><i style="width:{rates.get(k, 0) * 100:.1f}%;background:{c}"></i></div></div>'
        f'<b>{_pct(rates.get(k, float("nan")), 0)}</b></li>' for k, name, letter, c in rows)
    st.html(f'<div class="dk-h"><h3>Hallazgo de imagen</h3><p>Proporción con etiqueta 1 en cada grupo</p></div><ul class="dk-prog">{items}</ul>')


def sex_card(data: pd.DataFrame) -> None:
    valid = data.riesgo_oncologico.isin([0, 1])
    total = max(len(data), 1)
    rates = data.loc[valid].groupby("sexo")["riesgo_oncologico"].mean()
    blocks = [("F", "Mujeres", "#e3f4ec", GREEN), ("M", "Hombres", "#f3f9e3", "#6f8f17")]
    inner = "".join(f'<div style="background:{bg}"><small style="color:{fg}">{name}</small><b>{_pct((data.sexo == k).sum() / total)}</b>'
                    f'<span>etiqueta 1 · {_pct(rates.get(k, float("nan")))}</span></div>' for k, name, bg, fg in blocks)
    st.html(f'<div class="dk-h"><h3>Sexo</h3><p>Distribución y etiqueta 1</p></div><div class="dk-sex">{inner}</div>')


# ---------------- infografía de factores de riesgo ----------------
FACTORS = [  # clave, nombre, letra, lado, (x, y) del punto en la silueta, ícono
    ("familiar", "Antecedente familiar", "A", "izq", (100, 34), "dna"),
    ("tabaco", "Tabaquismo activo", "B", "izq", (86, 112), "lung"),
    ("imagen", "Imagen sospechosa", "C", "izq", (114, 128), "scan"),
    ("obesidad", "Obesidad", "D", "der", (100, 178), "scale"),
    ("benigno", "Antecedente benigno", "E", "der", (118, 160), "shield"),
    ("ca125", "CA-125 elevado", "F", "der", (100, 214), "drop"),
]
ICONS = {
    "dna": '<path d="M7 3c0 6 10 6 10 12s-10 6-10 6M17 3c0 6-10 6-10 12M8 7h8M8 17h8"/>',
    "lung": '<path d="M12 4v8M12 12c-2-1-3-4-5-4-2 0-3 4-3 8 0 3 2 4 4 4 3 0 4-2 4-5zM12 12c2-1 3-4 5-4 2 0 3 4 3 8 0 3-2 4-4 4-3 0-4-2-4-5z"/>',
    "scan": '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3.5"/>',
    "scale": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8.5 9a5 5 0 0 1 7 0L12 12z"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    "drop": '<path d="M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11z"/>',
}
SILHOUETTE = ('<path fill="#dfe8e3" d="M100 8c13 0 22 10 22 23s-9 25-22 25-22-12-22-25S87 8 100 8zM84 60h32c18 0 30 10 33 26l10 64c1 7-9 9-11 2l-11-52'
              '-2 60 6 104c0 9-13 10-15 1l-13-90h-6l-13 90c-2 9-15 8-15-1l6-104-2-60-11 52c-2 7-12 5-11-2l10-64c3-16 15-26 33-26z"/>')


def _factor_masks(data: pd.DataFrame) -> dict:
    return {"familiar": data.antecedente_familiar_cancer == 1, "tabaco": data.tabaquismo == "activo",
            "imagen": data.resultado_imagen == "sospechoso", "obesidad": data.imc >= 30,
            "benigno": data.antecedente_personal_benigno == 1, "ca125": data.ca125_u_ml > 35}


def risk_infographic(data: pd.DataFrame) -> None:
    """Puntos unidos: % de etiqueta 1 sin el factor (gris) y con el factor (naranja)."""
    valid = data.riesgo_oncologico.isin([0, 1])
    d = data.loc[valid]
    masks = {k: m.loc[valid].fillna(False) for k, m in _factor_masks(data).items()}
    base = float(d.riesgo_oncologico.mean()) if len(d) else float("nan")
    rows = []
    for key, name, letter, side, xy, icon in FACTORS:
        m = masks[key]
        with_f, without = d.loc[m, "riesgo_oncologico"], d.loc[~m, "riesgo_oncologico"]
        if with_f.empty or without.empty:
            continue
        rows.append(dict(name=name, icon=icon, share=float(m.mean()), con=float(with_f.mean()), sin=float(without.mean()), n=int(m.sum())))
    if not rows or np.isnan(base):
        return
    rows.sort(key=lambda r: r["con"] - r["sin"], reverse=True)
    top = max(max(r["con"], r["sin"]) for r in rows)
    scale = min(1.0, np.ceil(top * 10 + .5) / 10)          # eje hasta la decena siguiente
    pos = lambda v: f"{v / scale * 100:.2f}%"
    ticks = "".join(f'<span style="left:{pos(t)}">{int(round(t * 100))} %</span>' for t in np.arange(0, scale + 1e-9, .1))
    body = ""
    for r in rows:
        icon = _svg_img(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="1.8" '
                        f'stroke-linecap="round" stroke-linejoin="round">{ICONS[r["icon"]]}</svg>', "dk-fsvg")
        lo, hi = sorted((r["sin"], r["con"]))
        up = r["con"] >= r["sin"]
        body += (f'<div class="db-row"><div class="db-name"><span class="dk-fi">{icon}</span><div><b>{html.escape(r["name"])}</b>'
                 f'<small>{_pct(r["share"], 0)} de la selección</small></div></div>'
                 f'<div class="db-track"><i class="db-line" style="left:{pos(lo)};width:calc({pos(hi)} - {pos(lo)})"></i>'
                 f'<i class="db-dot sin" style="left:{pos(r["sin"])}"><em>{_pct(r["sin"], 0)}</em></i>'
                 f'<i class="db-dot con{"" if up else " baja"}" style="left:{pos(r["con"])}"><em>{_pct(r["con"], 0)}</em></i></div></div>')
    best = rows[0]
    st.html(f'''<div class="dk-info">
      <div class="dk-h"><h3>Factores asociados con la etiqueta 1</h3><p>Porcentaje de pacientes con etiqueta 1 entre quienes no tienen el factor y entre quienes sí lo tienen. Ordenados del que más sube al que menos.</p></div>
      <div class="db-legend"><span><i class="db-dot sin"></i>Sin el factor</span><span><i class="db-dot con"></i>Con el factor</span><span><i class="db-base-k"></i>Promedio de la selección: {_pct(base)}</span></div>
      <div class="db-chart"><div class="db-axis">{ticks}</div>{body}
        <i class="db-base" style="left:calc(var(--db-label) + (100% - var(--db-label)) * {base / scale:.4f})"></i></div>
      <p class="dk-note">Lectura: con {html.escape(best["name"].lower())}, la etiqueta 1 pasa de {_pct(best["sin"], 0)} a {_pct(best["con"], 0)}. Son asociaciones en datos sintéticos; no indican causalidad.</p></div>''')


def render_dashboard(data: pd.DataFrame, bundle) -> None:
    kpi_cards(data)
    left, right = st.columns([1.55, 1], gap="medium")
    with left, st.container(key="dk_trend"):
        trend_chart(data)
    with right, st.container(key="dk_model"):
        model_card(bundle)
    a, b, c = st.columns([1.15, 1, 1], gap="medium")
    with a, st.container(key="dk_age"):
        age_chart(data)
    with b, st.container(key="dk_image"):
        image_card(data)
    with c, st.container(key="dk_sex"):
        sex_card(data)
    with st.container(key="dk_info"):
        risk_infographic(data)
