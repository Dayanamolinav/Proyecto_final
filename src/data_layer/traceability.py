"""
Bitácora de trazabilidad (Fase 2/3 y Sección VII-A.8 "Gobernanza, seguridad
y trazabilidad" del informe).

Registra, en un archivo JSON-lines append-only bajo `logs/`, cada carga y
transformación aplicada sobre los datos: fecha, versión, regla/proceso
aplicado, campo afectado y número de registros modificados. Esto sustituye,
a escala de prototipo, la bitácora de transformaciones y el registro de
experimentos exigidos por la metodología.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LOGS_DIR

TRACE_LOG_PATH = LOGS_DIR / "bitacora_trazabilidad.jsonl"


@dataclass
class TraceEvent:
    stage: str  # p.ej. "INGESTA", "RAW", "CLEAN", "ANALYTICAL", "MODELADO"
    action: str  # p.ej. "carga_csv", "eliminacion_duplicados"
    details: dict[str, Any] = field(default_factory=dict)
    rows_affected: int | None = None
    version: str = "v1"

    def to_record(self) -> dict[str, Any]:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": self.stage,
            "action": self.action,
            "version": self.version,
            "rows_affected": self.rows_affected,
            "details": self.details,
        }


def log_event(event: TraceEvent, log_path: Path = TRACE_LOG_PATH) -> None:
    """Agrega un evento a la bitácora (append-only) para trazabilidad."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")


def read_trace_log(log_path: Path = TRACE_LOG_PATH) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
