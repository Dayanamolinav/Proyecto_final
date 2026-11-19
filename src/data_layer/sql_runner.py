"""
Ejecución de los scripts de sql/ en Snowflake (arquitectura VII-B.4).

Las transformaciones RAW -> CLEAN -> ANALYTICAL existen en dos versiones
equivalentes: SQL (sql/02 y sql/03, para ejecutarse dentro de Snowflake) y
Python (src/processing, para trabajar en local sin conexión). La prueba
tests/test_sql_transformations.py comprueba que ambas producen las mismas filas.
"""
from __future__ import annotations

from pathlib import Path

from config import BASE_DIR

SQL_DIR = BASE_DIR / "sql"
CLEAN_SQL = SQL_DIR / "02_raw_a_clean.sql"
ANALYTICAL_SQL = SQL_DIR / "03_clean_a_analytical.sql"


def read_sql_statements(path: str | Path) -> list[str]:
    """Devuelve las sentencias de un archivo .sql, sin comentarios de línea.

    Los scripts del proyecto no usan ';' dentro de textos ni comentarios, así
    que basta con separar por ';'."""
    lines = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    statements = [s.strip() for s in "\n".join(lines).split(";")]
    return [s for s in statements if s]


def run_sql_file(connection, path: str | Path) -> list[list[tuple]]:
    """Ejecuta cada sentencia del archivo con una conexión de Snowflake y
    devuelve las filas de las consultas que producen resultados."""
    results = []
    with connection.cursor() as cur:
        database = connection.database or "ONCORISK_DB"
        cur.execute(f"USE DATABASE {database}")
        for statement in read_sql_statements(path):
            cur.execute(statement)
            try:
                results.append(cur.fetchall())
            except Exception:  # sentencias sin conjunto de resultados
                results.append([])
    return results
