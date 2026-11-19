#!/usr/bin/env python
"""
CLI para ejecutar el pipeline completo del prototipo.

Uso:
    python scripts/run_pipeline.py --backend local --usar-csv-local      # sin conexión: los 20.000 sintéticos de data/synthetic/
    python scripts/run_pipeline.py --backend local                       # lee los 20.000 de Snowflake, guarda capas en local
    python scripts/run_pipeline.py --backend snowflake                   # todo contra Snowflake (pide el token PAT)
    python scripts/run_pipeline.py --backend snowflake --transformar-con-sql  # CLEAN y ANALYTICAL con sql/02 y sql/03
    python scripts/run_pipeline.py --backend local --usar-datos-locales  # lote sintético pequeño generado al momento

Si no se indica --backend, se usa DATA_BACKEND del .env (o "local" si no existe).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_full_pipeline  # noqa: E402


def ensure_snowflake_credentials(need_connection: bool) -> None:
    """Si vamos a tocar Snowflake (para guardar capas o para leer el CSV de
    prueba), pide la contraseña o el token de forma oculta si no están ya en
    el entorno, antes de arrancar el pipeline."""
    if not need_connection:
        return
    authenticator = os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake").strip()
    if authenticator.lower() in {"snowflake", "username_password_mfa"} and not os.getenv("SNOWFLAKE_PASSWORD"):
        os.environ["SNOWFLAKE_PASSWORD"] = getpass.getpass(
            "Contraseña de Snowflake (no se mostrará ni guardará): "
        )
    elif authenticator.upper() == "PROGRAMMATIC_ACCESS_TOKEN" and not os.getenv("SNOWFLAKE_TOKEN"):
        os.environ["SNOWFLAKE_TOKEN"] = getpass.getpass(
            "Token PAT (no se mostrará ni guardará): "
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline de ingeniería de datos e IA para riesgo oncológico.")
    parser.add_argument("--csv", type=str, default=None, help="Ruta a un archivo CSV real (deshabilitado por ahora).")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["local", "snowflake"],
        help="Dónde escribir RAW/CLEAN/ANALYTICAL. Por defecto usa DATA_BACKEND del .env (local).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--usar-datos-locales",
        action="store_true",
        help="Genera un lote sintético local pequeño en vez de leer los 20.000 pacientes de Snowflake (no requiere conexión).",
    )
    source.add_argument(
        "--usar-csv-local",
        action="store_true",
        help="Usa la copia local de los mismos 20.000 pacientes sintéticos (data/synthetic/) sin conectarse a Snowflake.",
    )
    parser.add_argument(
        "--transformar-con-sql",
        action="store_true",
        help="Construye CLEAN y ANALYTICAL dentro de Snowflake con sql/02 y sql/03 (requiere --backend snowflake).",
    )
    args = parser.parse_args()

    effective_backend = args.backend or os.getenv("DATA_BACKEND", "local")
    if args.transformar_con_sql and effective_backend != "snowflake":
        parser.error("--transformar-con-sql requiere --backend snowflake")
    use_local_csv = args.usar_csv_local
    use_snowflake_csv = not (args.usar_datos_locales or use_local_csv)
    needs_snowflake = effective_backend == "snowflake" or use_snowflake_csv
    ensure_snowflake_credentials(needs_snowflake)

    origin = "csv_local" if use_local_csv else ("snowflake" if use_snowflake_csv else "generador_local")
    print(f"Ejecutando pipeline completo (backend={effective_backend}, origen={origin})...")
    summary = run_full_pipeline(csv_path=args.csv, backend=effective_backend, use_snowflake_csv=use_snowflake_csv,
                                use_local_synthetic_csv=use_local_csv, transform_with_sql=args.transformar_con_sql)
   
    print("\n=== Resumen ===")
    print(f"Modelo seleccionado: {summary['modelo_seleccionado']}")
    print(f"Registros en ANALYTICAL: {summary['n_analytical']}")
    print("Validación cruzada (AUROC medio por modelo):")
    for name, scores in summary["resultados_validacion_cruzada"].items():
        print(f"  - {name}: {scores['cv_mean']:.4f} (+/- {scores['cv_std']:.4f})")

    umbral = summary["umbral_seleccionado"]
    print(f"\nUmbral elegido en desarrollo (fuera de pliegue, máx. F1): {umbral['umbral']:.2f}")
    print("\nMétricas del modelo seleccionado sobre el conjunto de prueba reservado:")
    print(json.dumps(summary["evaluacion_prueba_reservada"]["modelo"], ensure_ascii=False, indent=2))

    print(f"\nModelo persistido en: {summary['ruta_modelo']}")
    print("Reportes en: reports/  |  Bitácora de trazabilidad en: logs/bitacora_trazabilidad.jsonl")


if __name__ == "__main__":
    main()