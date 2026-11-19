"""Conexión segura a Snowflake mediante variables de entorno."""
from __future__ import annotations
<<<<<<< HEAD
=======
from dotenv import load_dotenv
load_dotenv()
>>>>>>> 00e19cf546b21c5570e600468f7a8d417360b312

import os
from contextlib import contextmanager
from typing import Iterator


def _value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def connection_parameters() -> dict[str, str | bool]:
    required = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER")
    missing = [name for name in required if not _value(name)]
    if missing:
        raise RuntimeError("Faltan variables de entorno: " + ", ".join(missing))

    authenticator = _value("SNOWFLAKE_AUTHENTICATOR") or "PROGRAMMATIC_ACCESS_TOKEN"
    parameters: dict[str, str | bool] = {
        "account": _value("SNOWFLAKE_ACCOUNT"),
        "user": _value("SNOWFLAKE_USER"),
        "authenticator": authenticator,
        "application": "ONCORISK_ACADEMIC_PROTOTYPE",
        "enable_server_session_keep_alive_auto_detection": False,
    }
    optional = {
        "role": _value("SNOWFLAKE_ROLE"),
        "warehouse": _value("SNOWFLAKE_WAREHOUSE"),
        "database": _value("SNOWFLAKE_DATABASE"),
        "schema": _value("SNOWFLAKE_SCHEMA"),
        "password": _value("SNOWFLAKE_PASSWORD"),
        "token": _value("SNOWFLAKE_TOKEN"),
    }
    parameters.update({key: value for key, value in optional.items() if value})
    if authenticator in {"snowflake", "username_password_mfa"} and "password" not in parameters:
        raise RuntimeError("Falta SNOWFLAKE_PASSWORD para autenticación con contraseña.")
    if authenticator.upper() == "PROGRAMMATIC_ACCESS_TOKEN" and "token" not in parameters:
        raise RuntimeError("Falta SNOWFLAKE_TOKEN para autenticación PAT.")
    return parameters


@contextmanager
def snowflake_connection() -> Iterator[object]:
    try:
        import snowflake.connector
    except ImportError as exc:
        raise RuntimeError(
            "Falta snowflake-connector-python. Instale requirements.txt."
        ) from exc

    connection = snowflake.connector.connect(**connection_parameters())
    try:
        yield connection
    finally:
        connection.close()
