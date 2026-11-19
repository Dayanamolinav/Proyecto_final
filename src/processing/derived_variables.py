"""Definiciones compartidas entre entrenamiento y formulario."""
import pandas as pd


def derive_variables(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "imc" in result.columns:
        result["imc_categoria"] = pd.cut(
            result["imc"], bins=[0, 18.5, 25, 30, 101],
            labels=["bajo_peso", "normal", "sobrepeso", "obesidad"], right=False,
        ).astype(str)
    if "edad" in result.columns:
        result["grupo_etario"] = pd.cut(
            result["edad"], bins=[0, 40, 55, 70, 121],
            labels=["<40", "40-54", "55-69", "70+"], right=False,
        ).astype(str)
    return result
