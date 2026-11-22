"""Carga validada; reemplazo transaccional de la tabla de entrada."""
from uuid import uuid4
from src.data_layer.ingestion import read_uploaded_csv, validate_uploaded_frame, REQUIRED_UPLOAD_COLUMNS
from src.data_layer.snowflake_connection import snowflake_connection

def replace_uploaded_csv(path):
    from snowflake.connector.pandas_tools import write_pandas
    frame = read_uploaded_csv(path)
    problems = validate_uploaded_frame(frame)
    if problems:
        raise ValueError(" ".join(problems))
    frame = frame[REQUIRED_UPLOAD_COLUMNS].copy()
    frame.columns = [c.upper() for c in frame.columns]
    temporary = "UPLOAD_" + uuid4().hex.upper()
    with snowflake_connection() as conn, conn.cursor() as cur:
        cur.execute(f"CREATE TEMPORARY TABLE RAW.{temporary} LIKE RAW.PACIENTES_PRUEBA_SINTETICOS")
        try:
            success, chunks, count, output = write_pandas(conn, frame, temporary, schema="RAW", auto_create_table=False)
            if not success or count != len(frame):
                raise RuntimeError("Carga incompleta; se conserva la tabla anterior.")
            columns = ", ".join(frame.columns)
            cur.execute("BEGIN")
            try:
                cur.execute(f"INSERT OVERWRITE INTO RAW.PACIENTES_PRUEBA_SINTETICOS ({columns}) SELECT {columns} FROM RAW.{temporary}")
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
        finally:
            cur.execute(f"DROP TABLE IF EXISTS RAW.{temporary}")
    return len(frame)
