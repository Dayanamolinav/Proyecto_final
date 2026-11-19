#!/usr/bin/env python
"""Prueba de solo lectura para la conexión de Snowflake."""
from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_layer.snowflake_connection import snowflake_connection


def main() -> None:
    authenticator = os.getenv(
        "SNOWFLAKE_AUTHENTICATOR", "PROGRAMMATIC_ACCESS_TOKEN"
    ).strip()
    if authenticator.upper() == "PROGRAMMATIC_ACCESS_TOKEN" and not os.getenv("SNOWFLAKE_TOKEN"):
        os.environ["SNOWFLAKE_TOKEN"] = getpass.getpass(
            "Token PAT (no se mostrará ni guardará): "
        )
    elif authenticator in {"snowflake", "username_password_mfa"} and not os.getenv("SNOWFLAKE_PASSWORD"):
        os.environ["SNOWFLAKE_PASSWORD"] = getpass.getpass(
            "Contraseña de Snowflake (no se mostrará ni guardará): "
        )

    with snowflake_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE(), "
                "CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()"
            )
            context = dict(zip(
                ["account", "user", "role", "warehouse", "database", "schema"],
                cursor.fetchone(),
            ))
            cursor.execute("SHOW WAREHOUSES")
            columns = [column[0].lower() for column in cursor.description]
            warehouses = [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()

    safe_warehouses = [
        {key: item.get(key) for key in ("name", "state", "size", "owner")}
        for item in warehouses
    ]
    result = {"context": context, "warehouses": safe_warehouses}
    report_path = Path(__file__).resolve().parents[1] / "reports" / "snowflake_connection_test.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("\nConexión exitosa.")
    print(json.dumps(result, indent=2, default=str))
    print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nNo fue posible conectar.", file=sys.stderr)
        for attribute in ("errno", "sqlstate", "msg"):
            value = getattr(exc, attribute, None)
            if value is not None:
                print(f"{attribute}: {value}", file=sys.stderr)
        if not getattr(exc, "msg", None):
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
