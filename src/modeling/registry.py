"""
Registro de experimentos (Sección VI-C, Fase 3 y Fase 5 del informe):
variables utilizadas, transformaciones, particiones, semilla, algoritmo,
configuraciones, tiempo de ejecución y resultados, para garantizar
reproducibilidad.

Se persiste como JSON-lines append-only en `reports/registro_experimentos.jsonl`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import REPORTS_DIR

EXPERIMENTS_LOG_PATH = REPORTS_DIR / "registro_experimentos.jsonl"


@dataclass
class ExperimentRecord:
    model_name: str
    random_seed: int
    cv_folds: int
    test_size: float
    numeric_features: list[str]
    categorical_features: list[str]
    hyperparameters: dict[str, Any]
    cv_metric_mean: float
    cv_metric_std: float
    test_metrics: dict[str, float]
    training_seconds: float
    n_train: int
    n_test: int
    usa_pesos_clase: bool = False
    metricas_fuera_de_pliegue: dict[str, float] = field(default_factory=dict)
    umbral: dict[str, Any] = field(default_factory=dict)
    origen_datos: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_experiment(record: ExperimentRecord, path: Path = EXPERIMENTS_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")


def read_experiments(path: Path = EXPERIMENTS_LOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
