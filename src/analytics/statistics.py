"""Cálculos reproducibles del panel estadístico, independientes de Streamlit.

El archivo sintético representa una fila por paciente. Las estimaciones e
inferencias son ejercicios bajo supuestos de muestreo iid: no constituyen
estimaciones clínicas ni inferencia a una población hospitalaria. No se imputan
datos aquí. Cada cálculo comunica su denominador y conserva ausencias.
"""
from __future__ import annotations

import math
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = ["edad", "imc", "presion_sistolica", "glucosa_mg_dl", "paridad", "ca125_u_ml"]
CATEGORICAL_COLUMNS = ["sexo", "antecedente_familiar_cancer", "antecedente_personal_benigno", "tabaquismo", "resultado_imagen", "fuente", "periodo", "grupo_etario", "imc_categoria"]
TARGET = "riesgo_oncologico"
LABELS = {
    "id_paciente": "Paciente", "edad": "Edad (años)", "sexo": "Sexo",
    "imc": "IMC (kg/m²)", "presion_sistolica": "Presión sistólica (mmHg)",
    "glucosa_mg_dl": "Glucosa (mg/dL)", "paridad": "Paridad",
    "ca125_u_ml": "CA-125 (U/mL)", "tabaquismo": "Tabaquismo",
    "antecedente_familiar_cancer": "Antecedente familiar de cáncer",
    "antecedente_personal_benigno": "Antecedente personal benigno",
    "resultado_imagen": "Hallazgo de imagen", "fuente": "Procedencia",
    "periodo": "Periodo", TARGET: "Etiqueta sintética", "grupo_etario": "Grupo de edad",
    "imc_categoria": "Categoría de IMC",
}
_BOUNDS = {"edad": (18, 100), "imc": (12, 60), "presion_sistolica": (80, 220), "glucosa_mg_dl": (50, 350), "paridad": (0, 10), "ca125_u_ml": (0, 600)}
_CATEGORIES = {"sexo": {"F", "M"}, "antecedente_familiar_cancer": {0, 1}, "antecedente_personal_benigno": {0, 1}, "tabaquismo": {"nunca", "exfumador", "activo"}, "resultado_imagen": {"normal", "indeterminado", "sospechoso"}, "fuente": {"consulta_externa", "urgencias", "remision"}, TARGET: {0, 1}}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _target(df: pd.DataFrame) -> pd.Series:
    if TARGET not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = _numeric(df[TARGET])
    return values.where(values.isin([0, 1]))


def benjamini_hochberg(values) -> np.ndarray:
    """BH dentro de una familia; conserva NaN y el orden original."""
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    positions = np.flatnonzero(np.isfinite(values))
    if not len(positions):
        return result
    if np.any((values[positions] < 0) | (values[positions] > 1)):
        raise ValueError("Los valores p deben pertenecer a [0, 1].")
    order = positions[np.argsort(values[positions], kind="stable")]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    result[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """IC de Wilson para una proporción; devuelve fracciones, no porcentajes."""
    if not 0 < confidence < 1:
        raise ValueError("La confianza debe estar entre 0 y 1.")
    if total < 0 or successes < 0 or successes > total or int(total) != total or int(successes) != successes:
        raise ValueError("Se requiere 0 <= éxitos <= total con conteos enteros.")
    if total == 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen con DE/varianza muestral e IC t bilateral del 95 % de la media."""
    columns = ["variable", "n", "ausentes", "media", "mediana", "desv_est", "varianza", "minimo", "q1", "q3", "maximo", "iqr", "cv_pct", "asimetria", "curtosis_exceso", "ic_inf", "ic_sup"]
    rows = []
    for column in NUMERIC_COLUMNS:
        if column not in df:
            continue
        x = _numeric(df[column]).dropna()
        n = len(x)
        row = {key: math.nan for key in columns}
        row.update(variable=column, n=n, ausentes=len(df) - n)
        if n:
            mean = float(x.mean())
            q1, q3 = x.quantile([0.25, 0.75])
            row.update(media=mean, mediana=float(x.median()), minimo=float(x.min()), q1=float(q1), q3=float(q3), maximo=float(x.max()), iqr=float(q3 - q1))
            if n > 1:
                sd = float(x.std(ddof=1))
                half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
                row.update(desv_est=sd, varianza=sd * sd, cv_pct=100 * sd / abs(mean) if mean else math.nan, ic_inf=mean - half, ic_sup=mean + half)
                if sd > 0:
                    row["asimetria"] = float(stats.skew(x, bias=False)) if n > 2 else math.nan
                    row["curtosis_exceso"] = float(stats.kurtosis(x, fisher=True, bias=False)) if n > 3 else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Frecuencias con ausencias explícitas y denominador igual a filas filtradas."""
    if column not in df or df.empty:
        return pd.DataFrame(columns=["categoria", "n", "porcentaje"])
    values = df[column].astype(object).where(df[column].notna(), "Sin dato")
    counts = values.value_counts(dropna=False, sort=False)
    out = pd.DataFrame({"categoria": counts.index.map(str), "n": counts.to_numpy(dtype=int)})
    out["porcentaje"] = 100 * out["n"] / len(df)
    return out.sort_values("n", ascending=False, kind="stable").reset_index(drop=True)


def group_rates(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Proporción de etiqueta 1 por grupo y Wilson 95 %, expresados en %."""
    columns = [group, "n_total", "n", "positivos", "porcentaje", "ic_inf", "ic_sup"]
    if group not in df or df.empty:
        return pd.DataFrame(columns=columns)
    work = pd.DataFrame({group: df[group].astype(object).where(df[group].notna(), "Sin dato"), "_y": _target(df)})
    rows = []
    for name, part in work.groupby(group, dropna=False, observed=True, sort=False):
        y = part["_y"].dropna()
        n, positive = len(y), int(y.sum())
        low, high = wilson_interval(positive, n)
        rows.append({group: str(name), "n_total": len(part), "n": n, "positivos": positive, "porcentaje": 100 * positive / n if n else math.nan, "ic_inf": 100 * low, "ic_sup": 100 * high})
    return pd.DataFrame(rows, columns=columns)


def correlation_matrix(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    if method not in {"spearman", "pearson"}:
        raise ValueError("Método no soportado.")
    numeric = pd.DataFrame({c: _numeric(df[c]) for c in NUMERIC_COLUMNS if c in df})
    return numeric.corr(method=method, min_periods=3)


def correlation_table(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    if method not in {"spearman", "pearson"}:
        raise ValueError("Método no soportado.")
    columns = ["variable_x", "variable_y", "n", "coeficiente", "p", "q", "metodo"]
    rows = []
    for left, right in combinations([c for c in NUMERIC_COLUMNS if c in df], 2):
        pair = pd.DataFrame({"x": _numeric(df[left]), "y": _numeric(df[right])}).dropna()
        if len(pair) < 3 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
            continue
        test = stats.spearmanr(pair.x, pair.y) if method == "spearman" else stats.pearsonr(pair.x, pair.y)
        rows.append({"variable_x": left, "variable_y": right, "n": len(pair), "coeficiente": float(test.statistic), "p": float(test.pvalue), "metodo": method})
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        out["q"] = benjamini_hochberg(out.p)
    return out


def compare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Grupo etiqueta 1 menos 0. Welch, Mann–Whitney y tamaños de efecto.

    BH se aplica por separado a las seis pruebas Welch y las seis Mann–Whitney.
    No se elige el test por el valor p observado. Los IC son puntuales del 95 %,
    no intervalos simultáneos ajustados por comparaciones múltiples.
    """
    columns = ["variable", "n0", "n1", "media0", "media1", "mediana0", "mediana1", "diferencia_media", "ic_inf", "ic_sup", "t_welch", "p_welch", "q_welch", "u_mannwhitney", "p_mannwhitney", "q_mannwhitney", "hedges_g", "rank_biserial", "estado"]
    y = _target(df)
    rows = []
    for column in NUMERIC_COLUMNS:
        if column not in df:
            continue
        x = _numeric(df[column])
        g0, g1 = x[y.eq(0)].dropna(), x[y.eq(1)].dropna()
        n0, n1 = len(g0), len(g1)
        row = {key: math.nan for key in columns}
        row.update(variable=column, n0=n0, n1=n1, estado="Muestra insuficiente")
        if n0:
            row.update(media0=float(g0.mean()), mediana0=float(g0.median()))
        if n1:
            row.update(media1=float(g1.mean()), mediana1=float(g1.median()))
        if n0 and n1:
            row["diferencia_media"] = float(g1.mean() - g0.mean())
        if min(n0, n1) >= 2:
            row["estado"] = "Calculable; comparar magnitud e incertidumbre"
            v0, v1 = float(g0.var(ddof=1)), float(g1.var(ddof=1))
            a, b = v0 / n0, v1 / n1
            se = math.sqrt(a + b)
            if se > 0:
                degrees = (a + b) ** 2 / (a * a / (n0 - 1) + b * b / (n1 - 1))
                row["t_welch"] = row["diferencia_media"] / se
                row["p_welch"] = float(2 * stats.t.sf(abs(row["t_welch"]), degrees))
                half = float(stats.t.ppf(0.975, degrees)) * se
                row.update(ic_inf=row["diferencia_media"] - half, ic_sup=row["diferencia_media"] + half)
            elif row["diferencia_media"] == 0:
                row.update(t_welch=0.0, p_welch=1.0, ic_inf=0.0, ic_sup=0.0, estado="Ambos grupos constantes e iguales")
            else:
                row["estado"] = "Varianza nula: Welch e IC no definidos"
            pooled = math.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
            if pooled > 0:
                degrees = n0 + n1 - 2
                correction = math.exp(float(gammaln(degrees / 2) - gammaln((degrees - 1) / 2))) / math.sqrt(degrees / 2)
                row["hedges_g"] = correction * row["diferencia_media"] / pooled
            mw = stats.mannwhitneyu(g1, g0, alternative="two-sided", method="asymptotic")
            row.update(u_mannwhitney=float(mw.statistic), p_mannwhitney=float(mw.pvalue), rank_biserial=2 * float(mw.statistic) / (n0 * n1) - 1)
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        out["q_welch"] = benjamini_hochberg(out.p_welch)
        out["q_mannwhitney"] = benjamini_hochberg(out.p_mannwhitney)
    return out


def categorical_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Asociación con etiqueta: chi² Pearson; Fisher para 2×2 escasas."""
    columns = ["variable", "n", "metodo", "estadistico", "p", "q", "cramers_v", "min_esperado", "fraccion_esperada_menor5", "estado"]
    rows = []
    y = _target(df)
    for column in CATEGORICAL_COLUMNS:
        if column not in df:
            continue
        complete = pd.DataFrame({"x": df[column], "y": y}).dropna()
        table = pd.crosstab(complete.x, complete.y, dropna=True)
        row = {key: math.nan for key in columns}
        row.update(variable=column, n=len(complete), metodo="No calculable", estado="Se requieren dos categorías y ambas etiquetas")
        if min(table.shape, default=0) >= 2:
            chi = stats.chi2_contingency(table.to_numpy(), correction=False)
            expected = chi.expected_freq
            low = float((expected < 5).mean())
            row.update(metodo="Chi² Pearson", estadistico=float(chi.statistic), p=float(chi.pvalue), cramers_v=math.sqrt(float(chi.statistic) / (table.to_numpy().sum() * min(table.shape[0] - 1, table.shape[1] - 1))), min_esperado=float(expected.min()), fraccion_esperada_menor5=low, estado="Esperados compatibles con aproximación chi²")
            if table.shape == (2, 2) and low:
                fisher = stats.fisher_exact(table.to_numpy(), alternative="two-sided")
                row.update(metodo="Fisher exacta 2×2", estadistico=float(fisher.statistic), p=float(fisher.pvalue), estado="Estadístico Fisher = odds ratio; V usa chi² sin corrección")
            elif (expected < 1).any() or low > 0.2:
                row["estado"] = "Escasez de frecuencias: p chi² aproximado poco fiable"
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        out["q"] = benjamini_hochberg(out.p)
    return out


def quality_report(df: pd.DataFrame) -> dict:
    """Perfil de la selección sin borrar, imputar ni corregir observaciones."""
    n_rows = len(df)
    total_cells = df.size
    missing = int(df.isna().sum().sum())
    duplicate_ids = int(df.id_paciente.dropna().duplicated().sum()) if "id_paciente" in df else 0
    summary = {"rows": n_rows, "patients": int(df.id_paciente.nunique()) if "id_paciente" in df else n_rows, "duplicate_rows": int(df.duplicated().sum()), "duplicate_ids": duplicate_ids, "missing_cells": missing, "missing_pct": 100 * missing / total_cells if total_cells else math.nan, "invalid_cells": 0, "contradictions": 0}
    columns, issues = [], []
    for column in df.columns:
        values = df[column]
        absent = int(values.isna().sum())
        invalid = pd.Series(False, index=df.index)
        outliers = 0
        if column in _BOUNDS:
            numeric = _numeric(values)
            low, high = _BOUNDS[column]
            invalid = values.notna() & (numeric.isna() | ~numeric.between(low, high))
            if column == "paridad":
                invalid |= numeric.notna() & numeric.mod(1).ne(0)
            q1, q3 = numeric.quantile([0.25, 0.75])
            iqr = q3 - q1
            if np.isfinite(iqr):
                outliers = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())
        elif column in _CATEGORIES:
            invalid = values.notna() & ~values.isin(_CATEGORIES[column])
        count_invalid = int(invalid.sum())
        summary["invalid_cells"] += count_invalid
        columns.append({"variable": column, "tipo": str(values.dtype), "n": int(values.notna().sum()), "ausentes": absent, "ausentes_pct": 100 * absent / n_rows if n_rows else math.nan, "unicos": int(values.nunique()), "invalidos": count_invalid, "atipicos_iqr": outliers})
        if absent:
            issues.append({"control": "Ausencias", "variable": column, "cantidad": absent, "porcentaje": 100 * absent / n_rows, "detalle": "Se conservan como ausentes; denominadores válidos por cálculo."})
        if count_invalid:
            issues.append({"control": "Dominio", "variable": column, "cantidad": count_invalid, "porcentaje": 100 * count_invalid / n_rows, "detalle": "Fuera de categorías o límites técnicos del prototipo; revisar sin borrar automáticamente."})
        if outliers:
            issues.append({"control": "Atípicos IQR", "variable": column, "cantidad": outliers, "porcentaje": 100 * outliers / n_rows, "detalle": "Fuera de Q1−1,5·IQR o Q3+1,5·IQR; no implica error ni exclusión."})
    if {"sexo", "paridad"}.issubset(df.columns):
        contradictions = int((df.sexo.eq("M") & _numeric(df.paridad).gt(0)).sum())
        summary["contradictions"] = contradictions
        if contradictions:
            issues.append({"control": "Coherencia del generador", "variable": "sexo / paridad", "cantidad": contradictions, "porcentaje": 100 * contradictions / n_rows, "detalle": "El generador asignó paridad > 0 a registros sexo M. Revisar la definición del campo y su coherencia antes de interpretar paridad; no se excluyen registros."})
    for key, label in [("duplicate_rows", "Filas duplicadas"), ("duplicate_ids", "Identificadores repetidos")]:
        if summary[key]:
            issues.append({"control": label, "variable": "id_paciente" if key == "duplicate_ids" else "todas", "cantidad": summary[key], "porcentaje": 100 * summary[key] / n_rows, "detalle": "Conteo de repeticiones después de la primera; no se eliminan automáticamente."})
    return {"summary": summary, "columns": pd.DataFrame(columns, columns=["variable", "tipo", "n", "ausentes", "ausentes_pct", "unicos", "invalidos", "atipicos_iqr"]), "issues": pd.DataFrame(issues, columns=["control", "variable", "cantidad", "porcentaje", "detalle"])}


def _multivariate_inputs(df: pd.DataFrame):
    features = [c for c in NUMERIC_COLUMNS if c in df]
    numeric = pd.DataFrame({c: _numeric(df[c]) for c in features}, index=df.index)
    complete = numeric.dropna()
    varying = [c for c in features if complete[c].nunique() > 1]
    return complete[varying], varying


def _score_metadata(df: pd.DataFrame, selected_index) -> pd.DataFrame:
    result = pd.DataFrame({"row_index": selected_index}, index=selected_index)
    result["id_paciente"] = df.loc[selected_index, "id_paciente"].to_numpy() if "id_paciente" in df else selected_index
    result[TARGET] = _target(df).loc[selected_index].to_numpy()
    return result.reset_index(drop=True)


def pca_analysis(df: pd.DataFrame) -> dict:
    """PCA exploratorio sobre casos completos y variables estandarizadas."""
    # Un índice local único evita multiplicar filas si el llamante reutiliza índices.
    data = df.reset_index(drop=True)
    complete, features = _multivariate_inputs(data)
    result = {"scores": pd.DataFrame(), "variance": pd.DataFrame(columns=["componente", "varianza_pct", "acumulada_pct"]), "loadings": pd.DataFrame(), "n": len(complete), "excluded": len(data) - len(complete), "features": features, "status": "Se requieren al menos tres casos completos y dos variables no constantes."}
    if len(complete) < 3 or len(features) < 2:
        return result
    scaled = StandardScaler().fit_transform(complete)
    model = PCA(n_components=min(len(features), len(complete)), svd_solver="full")
    coordinates = model.fit_transform(scaled)
    names = [f"PC{i + 1}" for i in range(coordinates.shape[1])]
    scores = _score_metadata(data, complete.index)
    for i, name in enumerate(names):
        scores[name] = coordinates[:, i]
    variance = 100 * model.explained_variance_ratio_
    loadings = pd.DataFrame(model.components_.T, columns=names)
    loadings.insert(0, "variable", features)
    result.update(scores=scores, variance=pd.DataFrame({"componente": names, "varianza_pct": variance, "acumulada_pct": variance.cumsum()}), loadings=loadings, status="Calculado")
    return result


def clustering_analysis(df: pd.DataFrame, k: int = 3) -> dict:
    """K-means descriptivo, semilla fija; clusters no son estratos clínicos."""
    if not isinstance(k, int) or not 2 <= k <= 8:
        raise ValueError("El número de grupos debe ser entero entre 2 y 8.")
    data = df.reset_index(drop=True)
    complete, features = _multivariate_inputs(data)
    result = {"scores": pd.DataFrame(), "centroids": pd.DataFrame(), "summary": pd.DataFrame(columns=["cluster", "n", "proporcion_pct", "n_etiqueta", "etiqueta1_pct"]), "silhouette": math.nan, "n": len(complete), "excluded": len(data) - len(complete), "features": features, "status": "Se requieren más casos completos distintos que clusters y dos variables no constantes."}
    if len(features) < 2 or len(complete) <= k or len(complete.drop_duplicates()) <= k:
        return result
    scaler = StandardScaler()
    scaled = scaler.fit_transform(complete)
    model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    assigned = model.fit_predict(scaled)
    scores = _score_metadata(data, complete.index)
    scores["cluster"] = assigned.astype(int) + 1
    for column in features:
        scores[column] = complete[column].to_numpy()
    centroids = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)
    centroids.insert(0, "cluster", np.arange(1, k + 1))
    rows = []
    for number, group in scores.groupby("cluster", observed=True):
        y = group[TARGET].dropna()
        rows.append({"cluster": int(number), "n": len(group), "proporcion_pct": 100 * len(group) / len(scores), "n_etiqueta": len(y), "etiqueta1_pct": 100 * float(y.mean()) if len(y) else math.nan})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            silhouette = float(silhouette_score(scaled, assigned, sample_size=min(1000, len(scaled)), random_state=42))
        except ValueError:
            silhouette = math.nan
    result.update(scores=scores, centroids=centroids, summary=pd.DataFrame(rows), silhouette=silhouette, status="Calculado; silhouette aproximado con hasta 1.000 casos, semilla 42")
    return result


def density_curve(values, points: int = 128) -> pd.DataFrame:
    """KDE gaussiana (Scott), sin afirmar normalidad ni usarla para conteos."""
    x = _numeric(pd.Series(values)).dropna().to_numpy(dtype=float)
    columns = ["valor", "densidad"]
    if len(x) < 3 or np.std(x) <= 0:
        return pd.DataFrame(columns=columns)
    grid = np.linspace(float(x.min()), float(x.max()), points)
    try:
        density = stats.gaussian_kde(x, bw_method="scott")(grid)
    except (ValueError, np.linalg.LinAlgError):
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({"valor": grid, "densidad": density})


def normality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Shapiro-Wilk bilateral sobre hasta 5000 valores, semilla 42.

    No se usa KS con parámetros estimados (p no calibrado) ni Shapiro n>5000.
    Los contrastes describen forma, no deciden automáticamente otro test.
    """
    rows = []
    for column in NUMERIC_COLUMNS:
        if column not in df:
            continue
        values = _numeric(df[column]).dropna()
        sample = values.sample(n=min(5000, len(values)), random_state=42)
        row = {"variable": column, "n_total": len(values), "n_prueba": len(sample), "w": math.nan,
               "p": math.nan, "q": math.nan, "estado": "Se requieren 3 valores y variación"}
        if len(sample) >= 3 and sample.nunique() > 1:
            test = stats.shapiro(sample)
            row.update(w=float(test.statistic), p=float(test.pvalue), estado="Submuestra reproducible de 5.000" if len(values) > 5000 else "Todos los valores válidos")
        rows.append(row)
    out = pd.DataFrame(rows, columns=["variable", "n_total", "n_prueba", "w", "p", "q", "estado"])
    out["q"] = benjamini_hochberg(out.p)
    return out


def odds_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """OR crudas (no ajustadas) por categoría contra referencia explícita.

    a/c: etiqueta 1/0 en exposición; b/d: etiqueta 1/0 en referencia.
    OR=ad/bc. IC Wald log(OR) ±1.96√(1/a+1/b+1/c+1/d).
    Haldane-Anscombe +0.5 en todas las celdas si alguna es cero, marcado.
    """
    references = {"sexo": "F", "antecedente_familiar_cancer": 0, "antecedente_personal_benigno": 0,
                  "tabaquismo": "nunca", "resultado_imagen": "normal", "fuente": "consulta_externa",
                  "periodo": "2023-S1", "grupo_etario": "<40", "imc_categoria": "normal"}
    columns = ["variable", "categoria", "referencia", "n_categoria", "n_referencia", "or", "ic_inf", "ic_sup", "p_fisher", "q", "correccion", "estado"]
    rows = []
    target = _target(df)
    for column, reference in references.items():
        if column not in df:
            continue
        for value in sorted(df[column].dropna().unique(), key=str):
            if value == reference:
                continue
            exposed, base = target[df[column].eq(value)].dropna(), target[df[column].eq(reference)].dropna()
            row = {c: math.nan for c in columns}
            row.update(variable=column, categoria=value, referencia=reference, n_categoria=len(exposed), n_referencia=len(base), correccion="Ninguna", estado="Referencia o categoría sin datos")
            if len(exposed) and len(base):
                cells = np.array([int(exposed.sum()), int(base.sum()), int((exposed == 0).sum()), int((base == 0).sum())], dtype=float)
                row["p_fisher"] = float(stats.fisher_exact([[cells[0], cells[2]], [cells[1], cells[3]]]).pvalue)
                if (cells == 0).any():
                    cells += .5
                    row["correccion"] = "+0,5 en las 4 celdas"
                a, b, c, d = cells
                log_or = math.log(a) + math.log(d) - math.log(b) - math.log(c)
                half = float(stats.norm.ppf(.975)) * math.sqrt(float((1 / cells).sum()))
                row.update(or_=math.exp(log_or))
                row["or"] = row.pop("or_")
                row.update(ic_inf=math.exp(log_or - half), ic_sup=math.exp(log_or + half), estado="OR cruda; IC aproximado puntual")
            rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    out["q"] = benjamini_hochberg(out.p_fisher)
    return out


def temporal_summary(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Medias por semestre e IC t; móvil agrupando registros de 2 semestres.

    No interpola observaciones, no crea fechas de consulta; conserva huecos.
    La media móvil se pondera por n al combinar las filas originales.
    """
    import re
    periods = sorted(str(p) for p in df.periodo.dropna().unique() if re.fullmatch(r"\d{4}-S[12]", str(p)))
    columns = ["periodo", "n", "media", "ic_inf", "ic_sup", "n_movil", "media_movil", "movil_inf", "movil_sup", "tramo"]
    if not periods:
        return pd.DataFrame(columns=columns)
    ordinal = lambda p: int(p[:4]) * 2 + int(p[-1]) - 1
    all_periods = [f"{k // 2}-S{k % 2 + 1}" for k in range(ordinal(periods[0]), ordinal(periods[-1]) + 1)]
    rows = []
    segment = 0
    for i, period in enumerate(all_periods):
        if i and period in periods and all_periods[i - 1] not in periods:
            segment += 1
        row = dict.fromkeys(columns, math.nan)
        row["periodo"] = period
        row["tramo"] = segment
        current = df.loc[df.periodo.eq(period)]
        row["n"] = len(_numeric(current[variable]).dropna())
        if len(current):
            result = numeric_summary(current).set_index("variable").loc[variable]
            row.update(media=result.media, ic_inf=result.ic_inf, ic_sup=result.ic_sup)
        if i > 0 and period in periods and all_periods[i - 1] in periods:
            pooled = df.loc[df.periodo.isin(all_periods[i - 1:i + 1])]
            result = numeric_summary(pooled).set_index("variable").loc[variable]
            row.update(n_movil=int(result.n), media_movil=result.media, movil_inf=result.ic_inf, movil_sup=result.ic_sup)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
