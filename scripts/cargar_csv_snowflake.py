#!/usr/bin/env python
"""
Carga el CSV de pacientes a Snowflake por el Stage interno (arquitectura VII-B.2):

    1. PUT del archivo local a @RAW.STG_PACIENTES
    2. COPY INTO a una tabla temporal, registro en AUDITORIA.REGISTRO_CARGAS e
       INSERT OVERWRITE en RAW.PACIENTES_PRUEBA_SINTETICOS (si falla, no se toca)
       (sql/01_ingesta_stage.sql)

Uso:
    python scripts/cargar_csv_snowflake.py --setup     # primera vez: crea objetos (sql/00) y carga
    python scripts/cargar_csv_snowflake.py             # recarga el CSV sintético incluido
    python scripts/cargar_csv_snowflake.py --csv otra_ruta.csv

Solo debe usarse con el lote sintético. Un archivo real del hospital requiere
antes el esquema, la autorización y la protección de identificadores aprobados
(documentacion/plan_datos_reales.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import LOCAL_SYNTHETIC_CSV  # noqa: E402
from src.data_layer.snowflake_connection import ask_for_missing_secret, snowflake_connection  # noqa: E402
from src.data_layer.sql_runner import SQL_DIR, read_sql_statements  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sube el CSV de pacientes sintéticos a Snowflake por Stage.")
    parser.add_argument("--csv", type=Path, default=LOCAL_SYNTHETIC_CSV, help="Archivo CSV a cargar.")
    parser.add_argument("--setup", action="store_true", help="Ejecuta antes sql/00_setup_inicial.sql.")
    args = parser.parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        parser.error(f"No existe {csv_path}")

    ask_for_missing_secret()
    with snowflake_connection() as conn, conn.cursor() as cur:
        if args.setup:
            for statement in read_sql_statements(SQL_DIR / "00_setup_inicial.sql"):
                cur.execute(statement)
            print("Objetos creados o verificados (sql/00_setup_inicial.sql).")
        cur.execute("USE DATABASE ONCORISK_DB")
        cur.execute("USE SCHEMA RAW")
        uri = csv_path.as_posix().replace("'", "\\'")
        cur.execute(f"PUT 'file://{uri}' @RAW.STG_PACIENTES AUTO_COMPRESS=TRUE OVERWRITE=TRUE")
        print(f"Archivo subido al Stage: {csv_path.name}")
        for statement in read_sql_statements(SQL_DIR / "01_ingesta_stage.sql"):
            cur.execute(statement)
        columns = [d[0] for d in cur.description]
        print("Carga terminada:", ", ".join(f"{c}={v}" for c, v in zip(columns, cur.fetchone())))
        print("Registro de la carga en AUDITORIA.REGISTRO_CARGAS.")


if __name__ == "__main__":
    main()
