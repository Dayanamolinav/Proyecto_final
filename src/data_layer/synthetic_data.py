"""
Generador de datos clínicos SINTÉTICOS para el prototipo.

IMPORTANTE: el informe (Sección VI-B) establece que el conjunto real será
suministrado por el hospital universitario y que, mientras tanto, variables,
cantidad de registros y estructura del archivo están por definir. Como ese
archivo aún no está disponible, este módulo produce un conjunto de prueba
PLAUSIBLE (no real, no proveniente de pacientes) con una estructura típica
de ficha clínica oncológica, para poder construir y validar el prototipo de
punta a punta. Cuando el archivo CSV real llegue, basta con reemplazar la
llamada a `generate_synthetic_clinical_data()` por la lectura del archivo real
en `src/data_layer/ingestion.py`: el resto de la canalización no cambia.

Las variables se inspiran en los antecedentes citados en el Marco Teórico
del informe (demográficas, antecedentes familiares/reproductivos, hábitos,
biomarcadores y hallazgos de imagen; ver [10], [11], [13] del documento),
generalizadas a "riesgo oncológico" en lugar de un tipo específico de cáncer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SyntheticDataConfig, TARGET_COLUMN, PATIENT_ID_COLUMN


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_clinical_data(cfg: SyntheticDataConfig | None = None) -> pd.DataFrame:
    """Genera un DataFrame con una ficha clínica oncológica sintética.

    Columnas (ficha de caracterización simplificada):
      - id_paciente: identificador seudonimizado (no es un identificador directo)
      - nombre, documento_identidad: identificadores DIRECTOS simulados; se
        incluyen a propósito para que la etapa de ingesta/CLEAN demuestre su
        eliminación, tal como exige la Sección E (Consideraciones éticas).
      - edad, sexo
      - antecedente_familiar_cancer, antecedente_personal_benigno
      - tabaquismo, imc, presion_sistolica, glucosa_mg_dl
      - paridad (número de embarazos; variable reproductiva)
      - ca125_u_ml (biomarcador numérico de ejemplo)
      - resultado_imagen (hallazgo categórico: "normal", "indeterminado",
        "sospechoso")
      - fuente, periodo (metadatos de procedencia, ficha de caracterización)
      - riesgo_oncologico: variable objetivo binaria (0/1)

    También se agregan columnas POSTERIORES al desenlace (fecha de
    diagnóstico, tratamiento indicado) únicamente para que `cleaning.py`
    pueda demostrar la regla de "no fuga de información" eliminándolas antes
    de ANALYTICAL.
    """
    cfg = cfg or SyntheticDataConfig()
    rng = np.random.default_rng(cfg.random_seed)
    n = cfg.n_patients

    edad = rng.normal(55, 13, n).clip(18, 95)
    sexo = rng.choice(["F", "M"], size=n, p=[0.78, 0.22])
    antecedente_familiar = rng.binomial(1, 0.22, n)
    antecedente_personal_benigno = rng.binomial(1, 0.12, n)
    tabaquismo = rng.choice(["nunca", "exfumador", "activo"], size=n, p=[0.55, 0.25, 0.20])
    imc = rng.normal(27, 5, n).clip(15, 48)
    presion_sistolica = rng.normal(125, 16, n).clip(85, 200)
    glucosa = rng.normal(100, 22, n).clip(55, 300)
    paridad = rng.poisson(1.6, n).clip(0, 8)
    ca125 = rng.lognormal(mean=2.6, sigma=0.5, size=n).clip(2, 400)
    resultado_imagen = rng.choice(
        ["normal", "indeterminado", "sospechoso"], size=n, p=[0.62, 0.26, 0.12]
    )
    fuente = rng.choice(["consulta_externa", "urgencias", "remision"], size=n, p=[0.6, 0.15, 0.25])
    periodo = rng.choice(["2023-S1", "2023-S2", "2024-S1", "2024-S2"], size=n)

    # Construcción de un "logit" de riesgo con relaciones plausibles con
    # algunas variables, para que los modelos tengan señal real que aprender
    # (esto es sintético: no representa evidencia clínica validada).
    imagen_score = pd.Series(resultado_imagen).map(
        {"normal": 0.0, "indeterminado": 0.9, "sospechoso": 2.1}
    ).to_numpy()
    tabaquismo_score = pd.Series(tabaquismo).map(
        {"nunca": 0.0, "exfumador": 0.25, "activo": 0.55}
    ).to_numpy()

    logit = (
        -3.3
        + 0.025 * (edad - 55)
        + 0.9 * antecedente_familiar
        + 0.35 * antecedente_personal_benigno
        + tabaquismo_score
        + 0.02 * (imc - 27)
        + 0.012 * (ca125 - 20)
        + imagen_score
        + rng.normal(0, 0.6, n)  # ruido no explicado
    )
    prob = _sigmoid(logit)

    # Calibrar aproximadamente a la prevalencia deseada ajustando el intercepto
    target_logit_shift = np.log(cfg.prevalence / (1 - cfg.prevalence)) - np.log(
        prob.mean() / (1 - prob.mean())
    )
    prob = _sigmoid(logit + target_logit_shift)
    riesgo_oncologico = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            PATIENT_ID_COLUMN: [f"PAC-{i:05d}" for i in range(1, n + 1)],
            "nombre": [f"Paciente Sintetico {i}" for i in range(1, n + 1)],
            "documento_identidad": rng.integers(10_000_000, 99_999_999, n).astype(str),
            "edad": edad.round(1),
            "sexo": sexo,
            "antecedente_familiar_cancer": antecedente_familiar,
            "antecedente_personal_benigno": antecedente_personal_benigno,
            "tabaquismo": tabaquismo,
            "imc": imc.round(1),
            "presion_sistolica": presion_sistolica.round(0),
            "glucosa_mg_dl": glucosa.round(0),
            "paridad": paridad,
            "ca125_u_ml": ca125.round(2),
            "resultado_imagen": resultado_imagen,
            "fuente": fuente,
            "periodo": periodo,
            TARGET_COLUMN: riesgo_oncologico,
            # Columnas POSTERIORES al desenlace (solo para demostrar
            # exclusión por fuga de información; nunca deben entrar al modelo)
            "fecha_diagnostico_confirmado": pd.NaT,
            "tratamiento_indicado": np.where(
                riesgo_oncologico == 1,
                rng.choice(["quimioterapia", "cirugia", "seguimiento"], n),
                "no_aplica",
            ),
            "comentario_oncologo": np.where(
                riesgo_oncologico == 1, "pendiente_revision_especialista", ""
            ),
        }
    )

    # Inyección controlada de valores faltantes (simula heterogeneidad real)
    for col, rate in cfg.missing_rate.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    # Duplicados exactos simulados (para que la limpieza los detecte y reporte)
    n_dupes = max(1, int(0.01 * n))
    dupe_rows = df.sample(n=n_dupes, random_state=cfg.random_seed).copy()
    df = pd.concat([df, dupe_rows], ignore_index=True)

    # Observación clínicamente imposible simulada (edad negativa) para
    # validar las reglas de calidad de datos
    bad_idx = df.sample(n=max(1, int(0.002 * n)), random_state=cfg.random_seed + 1).index
    df.loc[bad_idx, "edad"] = -1

    return df.sample(frac=1.0, random_state=cfg.random_seed).reset_index(drop=True)


if __name__ == "__main__":
    data = generate_synthetic_clinical_data()
    print(data.shape)
    print(data.head())
