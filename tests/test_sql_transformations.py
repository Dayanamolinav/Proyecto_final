"""Las transformaciones SQL (sql/02, sql/03) deben producir lo mismo que Python.

Snowflake no está disponible en las pruebas, así que se ejecutan los mismos
archivos en SQLite (incluido en Python) con tres ajustes mínimos de dialecto:
CREATE OR REPLACE -> DROP + CREATE, CURRENT_TIMESTAMP() -> CURRENT_TIMESTAMP,
y esquemas como bases adjuntas. Se compara contra src/processing.
"""
import re
import sqlite3
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from config import SyntheticDataConfig
from src.data_layer.sql_runner import ANALYTICAL_SQL, CLEAN_SQL, SQL_DIR, read_sql_statements
from src.data_layer.storage import pseudonymize_direct_identifiers
from src.data_layer.synthetic_data import generate_synthetic_clinical_data
from src.processing.cleaning import run_cleaning_pipeline
from src.processing.feature_engineering import build_analytical_table

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CSV = ROOT / "data" / "synthetic" / "pacientes_sinteticos_20000.csv"


def to_sqlite(statement: str) -> list[str]:
    statement = statement.replace("CURRENT_TIMESTAMP()", "CURRENT_TIMESTAMP")
    match = re.match(r"CREATE OR REPLACE TABLE\s+(\S+)\s+AS", statement, flags=re.I)
    if match:
        return [f"DROP TABLE IF EXISTS {match.group(1)}", statement.replace("CREATE OR REPLACE TABLE", "CREATE TABLE", 1)]
    return [statement]


def run_sql_in_sqlite(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(":memory:")
    for schema in ("RAW", "CLEAN", "ANALYTICAL", "AUDITORIA"):
        conn.execute(f"ATTACH DATABASE ':memory:' AS {schema}")
    conn.execute("CREATE TABLE AUDITORIA.BITACORA_TRANSFORMACIONES (FECHA_REGISTRO TEXT DEFAULT CURRENT_TIMESTAMP, "
                 "VERSION TEXT, ETAPA TEXT, REGLA TEXT, CAMPO TEXT, REGISTROS_AFECTADOS INTEGER)")
    upload = raw.copy()
    upload.columns = [c.upper() for c in upload.columns]
    for column in upload.select_dtypes(include=["string", "object", "str"]).columns:
        upload[column] = upload[column].astype(object).where(upload[column].notna(), None)
    upload.to_sql("CLINICAL_RAW_TMP", conn, index=False)  # pandas no escribe en bases adjuntas
    conn.execute("CREATE TABLE RAW.CLINICAL_RAW AS SELECT * FROM main.CLINICAL_RAW_TMP")
    for path in (CLEAN_SQL, ANALYTICAL_SQL):
        for statement in read_sql_statements(path):
            for translated in to_sqlite(statement):
                conn.execute(translated)
    read = lambda q: pd.read_sql_query(q, conn).rename(columns=str.lower)
    return (read("SELECT * FROM CLEAN.CLINICAL_CLEAN"), read("SELECT * FROM ANALYTICAL.CLINICAL_ANALYTICAL"),
            read("SELECT REGLA, REGISTROS_AFECTADOS FROM AUDITORIA.BITACORA_TRANSFORMACIONES"))


def canonical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = pd.to_numeric(out[column]).astype(float).round(6)
        else:
            out[column] = out[column].astype(object).where(out[column].notna(), None).astype(str)
    return out.sort_values(columns).reset_index(drop=True)


class SqlTransformationTests(unittest.TestCase):
    def assert_same_as_python(self, raw: pd.DataFrame) -> pd.DataFrame:
        with mock.patch("src.processing.cleaning.log_event"), mock.patch("src.processing.feature_engineering.log_event"):
            clean_py = run_cleaning_pipeline(raw)
            analytical_py = build_analytical_table(clean_py)
        clean_sql, analytical_sql, log = run_sql_in_sqlite(raw)
        self.assertEqual(len(clean_sql), len(clean_py))
        self.assertEqual(list(analytical_sql.columns), list(analytical_py.columns))
        cols = list(analytical_py.columns)
        pd.testing.assert_frame_equal(canonical(analytical_sql, cols), canonical(analytical_py, cols))
        return log

    def test_lote_con_duplicados_faltantes_imposibles_y_fuga(self):
        raw = generate_synthetic_clinical_data(SyntheticDataConfig(n_patients=600))
        raw, _ = pseudonymize_direct_identifiers(raw)
        raw.loc[raw.index[:3], "riesgo_oncologico"] = np.nan  # registros sin objetivo
        self.assertTrue(raw.duplicated().any())
        log = self.assert_same_as_python(raw).set_index("regla")["registros_afectados"]
        self.assertEqual(int(log["eliminacion_duplicados_exactos"]), int(raw.duplicated().sum()))
        self.assertGreater(int(log["eliminacion_valores_clinicamente_imposibles"]), 0)
        self.assertEqual(int(log["eliminacion_registros_sin_variable_objetivo"]), 3)

    @unittest.skipUnless(LOCAL_CSV.exists(), "Falta el CSV sintético local")
    def test_lote_de_20000_pacientes(self):
        raw = pd.read_csv(LOCAL_CSV, dtype={"id_paciente": "string"})
        self.assert_same_as_python(raw)

    def test_scripts_de_snowflake_existen_y_no_estan_vacios(self):
        for name in ("00_setup_inicial.sql", "01_ingesta_stage.sql", "02_raw_a_clean.sql", "03_clean_a_analytical.sql"):
            self.assertTrue(read_sql_statements(SQL_DIR / name), name)


if __name__ == "__main__":
    unittest.main()
