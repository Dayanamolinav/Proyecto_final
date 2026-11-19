-- =====================================================================
-- OncoRisk · 01 · Ingesta del CSV por Stage (arquitectura VII-B.2)
--
-- Paso previo (subir el archivo al Stage). PUT no funciona en las hojas de
-- Snowsight: se ejecuta desde SnowSQL o, más fácil, con el script
--   python scripts/cargar_csv_snowflake.py
-- que hace el PUT y luego ejecuta este archivo. Equivalente en SnowSQL:
--   PUT 'file://C:/ruta/pacientes_sinteticos_20000.csv' @RAW.STG_PACIENTES AUTO_COMPRESS=TRUE OVERWRITE=TRUE
--
-- Carga segura: el archivo entra primero a una tabla temporal. Solo si
-- COPY INTO termina sin errores, INSERT OVERWRITE reemplaza el lote de la
-- tabla de aterrizaje en un único paso. Si algo falla, la tabla queda intacta.
-- =====================================================================

USE DATABASE ONCORISK_DB;

CREATE OR REPLACE TEMPORARY TABLE RAW.PACIENTES_CARGA_TMP LIKE RAW.PACIENTES_PRUEBA_SINTETICOS;

COPY INTO RAW.PACIENTES_CARGA_TMP
  FROM @RAW.STG_PACIENTES
  FILE_FORMAT = (FORMAT_NAME = 'RAW.FF_CSV_PACIENTES')
  PATTERN = '.*pacientes_sinteticos.*[.]csv([.]gz)?'
  ON_ERROR = 'ABORT_STATEMENT'
  FORCE = TRUE;

-- Registro de origen y momento de la carga (resultado del COPY anterior)
INSERT INTO AUDITORIA.REGISTRO_CARGAS
  (TABLA_DESTINO, ARCHIVO_ORIGEN, FILAS_LEIDAS, FILAS_CARGADAS, ESTADO, PRIMER_ERROR, FECHA_CARGA)
SELECT 'RAW.PACIENTES_PRUEBA_SINTETICOS', "file", "rows_parsed", "rows_loaded", "status", "first_error", CURRENT_TIMESTAMP()
FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

-- Reemplazo del lote en un solo paso (carga por lotes con sobreescritura controlada)
INSERT OVERWRITE INTO RAW.PACIENTES_PRUEBA_SINTETICOS
SELECT * FROM RAW.PACIENTES_CARGA_TMP;

-- Verificación rápida
SELECT COUNT(*) AS FILAS, COUNT(DISTINCT ID_PACIENTE) AS PACIENTES, AVG(RIESGO_ONCOLOGICO) AS PREVALENCIA
FROM RAW.PACIENTES_PRUEBA_SINTETICOS;
