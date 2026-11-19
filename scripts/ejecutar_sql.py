#!/usr/bin/env python
"""
Ejecuta uno o varios archivos de sql/ en Snowflake con la conexión del .env.

Uso:
    python scripts/ejecutar_sql.py sql/00_setup_inicial.sql
    python scripts/ejecutar_sql.py sql/02_raw_a_clean.sql sql/03_clean_a_analytical.sql
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_layer.snowflake_connection import ask_for_missing_secret, snowflake_connection  # noqa: E402
from src.data_layer.sql_runner import read_sql_statements  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta archivos SQL del proyecto en Snowflake.")
    parser.add_argument("archivos", nargs="+", type=Path, help="Archivos .sql en el orden de ejecución.")
    args = parser.parse_args()
    for path in args.archivos:
        if not path.is_file():
            parser.error(f"No existe {path}")

    ask_for_missing_secret()
    with snowflake_connection() as conn, conn.cursor() as cur:
        for path in args.archivos:
            print(f"\n=== {path}")
            for statement in read_sql_statements(path):
                first_line = statement.splitlines()[0][:90]
                cur.execute(statement)
                print(f"  ok  {first_line}")
                if statement.lstrip().upper().startswith("SELECT"):
                    columns = [d[0] for d in cur.description]
                    for row in cur.fetchall():
                        print("      " + ", ".join(f"{c}={v}" for c, v in zip(columns, row)))


if __name__ == "__main__":
    main()
