#!/usr/bin/env python
"""
CLI para ejecutar el pipeline completo del prototipo.

Uso:
    python scripts/run_pipeline.py                 # usa datos sintéticos de prueba
    python scripts/run_pipeline.py --csv ruta.csv   # usa un archivo CSV real del hospital
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_full_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline de ingeniería de datos e IA para riesgo oncológico.")
    parser.add_argument("--csv", type=str, default=None, help="Ruta a un archivo CSV real (opcional). Si se omite, se usan datos sintéticos de prueba.")
    args = parser.parse_args()

    print("Ejecutando pipeline completo (ingesta -> RAW -> CLEAN -> ANALYTICAL -> modelado -> evaluación)...")
    summary = run_full_pipeline(csv_path=args.csv)

    print("\n=== Resumen ===")
    print(f"Modelo seleccionado: {summary['modelo_seleccionado']}")
    print(f"Registros en ANALYTICAL: {summary['n_analytical']}")
    print("Validación cruzada (AUROC medio por modelo):")
    for name, scores in summary["resultados_validacion_cruzada"].items():
        print(f"  - {name}: {scores['cv_mean']:.4f} (+/- {scores['cv_std']:.4f})")

    print("\nMétricas del modelo seleccionado sobre el conjunto de prueba reservado:")
    print(json.dumps(summary["evaluacion_prueba_reservada"]["modelo"], ensure_ascii=False, indent=2))

    print(f"\nModelo persistido en: {summary['ruta_modelo']}")
    print("Reportes en: reports/  |  Bitácora de trazabilidad en: logs/bitacora_trazabilidad.jsonl")


if __name__ == "__main__":
    main()
