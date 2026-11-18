"""
Configuración central del prototipo.

Centraliza rutas, semillas aleatorias y parámetros del proyecto para que
todo el pipeline (ingesta, capas RAW/CLEAN/ANALYTICAL, modelado y la app)
use una única fuente de verdad, favoreciendo trazabilidad y reproducibilidad
(Fase 3 y Sección D de la metodología del informe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
ANALYTICAL_DIR = DATA_DIR / "analytical"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

for _dir in (RAW_DIR, CLEAN_DIR, ANALYTICAL_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# Nombre de la variable objetivo tal como se documentará en el diccionario de
# datos (Fase 2). En este prototipo, al no existir fecha índice ni horizonte
# temporal confirmado, la variable se interpreta como clasificación
# probabilística de la condición registrada (riesgo_oncologico = 1/0),
# consistente con el Marco Conceptual del informe (sección V-C).
TARGET_COLUMN = "riesgo_oncologico"
PATIENT_ID_COLUMN = "id_paciente"

# Identificadores directos que deben excluirse/seudonimizarse antes de CLEAN
DIRECT_IDENTIFIERS = ["nombre", "documento_identidad"]

# Variables generadas después del "diagnóstico" que deben excluirse del
# modelado para evitar fuga de información (ver Marco Conceptual, "Fuga de
# información").
POST_OUTCOME_LEAKY_COLUMNS = [
    "fecha_diagnostico_confirmado",
    "tratamiento_indicado",
    "comentario_oncologo",
]

TEST_SIZE = 0.20
CV_FOLDS = 5

MODEL_CANDIDATES = ["logistic_regression", "random_forest", "xgboost"]

# Umbral de alta ausencia a partir del cual una variable se revisa con más
# cuidado antes de decidir su tratamiento (imputación/eliminación), según la
# Fase 2 (plan de calidad) del informe.
HIGH_MISSINGNESS_THRESHOLD = 0.40


@dataclass
class SyntheticDataConfig:
    """Parámetros del generador de datos de prueba (no son datos reales)."""

    n_patients: int = 1200
    prevalence: float = 0.28  # proporción aproximada de casos positivos
    random_seed: int = RANDOM_SEED
    missing_rate: dict = field(
        default_factory=lambda: {
            "imc": 0.05,
            "presion_sistolica": 0.08,
            "glucosa_mg_dl": 0.10,
            "ca125_u_ml": 0.15,
            "resultado_imagen": 0.12,
        }
    )
