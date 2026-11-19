"""Definiciones compartidas entre entrenamiento y formulario."""
import pandas as pd

# Un valor faltante de edad o IMC debe seguir siendo faltante en la variable
# derivada (luego se imputa con la moda del conjunto de desarrollo). Según la
# versión de pandas, astype(str) lo convertía en el texto 'nan' o en <NA>; se
# normaliza para que el resultado no dependa de la versión instalada.
_MISSING_TEXT = {"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA}


def _cut(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    result = pd.cut(values, bins=bins, labels=labels, right=False).astype(object)
    return result.where(result.notna(), pd.NA).astype(str).replace(_MISSING_TEXT).astype(object)


def derive_variables(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "imc" in result.columns:
        result["imc_categoria"] = _cut(result["imc"], [0, 18.5, 25, 30, 101],
                                       ["bajo_peso", "normal", "sobrepeso", "obesidad"])
    if "edad" in result.columns:
        result["grupo_etario"] = _cut(result["edad"], [0, 40, 55, 70, 121], ["<40", "40-54", "55-69", "70+"])
    return result
