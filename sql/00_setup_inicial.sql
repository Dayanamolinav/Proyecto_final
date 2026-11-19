-- =====================================================================
-- OncoRisk · 00 · Preparación de Snowflake (se ejecuta una sola vez)
-- Informe 2, Fase 2 y arquitectura física VII-B.2 y VII-B.3:
--   warehouse, base de datos, capas RAW / CLEAN / ANALYTICAL, esquema de
--   auditoría, formato de archivo, Stage interno y tabla de aterrizaje.
-- Puede ejecutarse en una hoja de Snowsight o con:
--   python scripts/ejecutar_sql.py sql/00_setup_inicial.sql
-- =====================================================================

CREATE WAREHOUSE IF NOT EXISTS ONCORISK_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'Cómputo del prototipo académico OncoRisk';

CREATE DATABASE IF NOT EXISTS ONCORISK_DB
  COMMENT = 'Prototipo académico OncoRisk (datos sintéticos)';

USE DATABASE ONCORISK_DB;

-- Capas del modelo de almacenamiento
CREATE SCHEMA IF NOT EXISTS RAW        COMMENT = 'Datos recibidos, sin identificadores directos';
CREATE SCHEMA IF NOT EXISTS CLEAN      COMMENT = 'Datos depurados y transformados';
CREATE SCHEMA IF NOT EXISTS ANALYTICAL COMMENT = 'Conjunto preparado para análisis y modelado';
CREATE SCHEMA IF NOT EXISTS AUDITORIA  COMMENT = 'Registro de cargas y bitácora de transformaciones';

-- Formato del CSV de entrada (encabezado, separador coma, nulos vacíos)
CREATE FILE FORMAT IF NOT EXISTS RAW.FF_CSV_PACIENTES
  TYPE = CSV
  FIELD_DELIMITER = ','
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  NULL_IF = ('', 'NULL', 'null', 'NaN')
  EMPTY_FIELD_AS_NULL = TRUE
  ENCODING = 'UTF8';

-- Stage interno por el que entra el archivo (referencia [18] del informe)
CREATE STAGE IF NOT EXISTS RAW.STG_PACIENTES
  FILE_FORMAT = RAW.FF_CSV_PACIENTES
  COMMENT = 'Archivos CSV de pacientes antes de COPY INTO';

-- Tabla de aterrizaje del lote sintético de 20.000 pacientes
CREATE TABLE IF NOT EXISTS RAW.PACIENTES_PRUEBA_SINTETICOS (
  ID_PACIENTE                  VARCHAR(20)   NOT NULL,
  EDAD                         FLOAT,
  SEXO                         VARCHAR(1),
  ANTECEDENTE_FAMILIAR_CANCER  NUMBER(1,0),
  ANTECEDENTE_PERSONAL_BENIGNO NUMBER(1,0),
  TABAQUISMO                   VARCHAR(20),
  IMC                          FLOAT,
  PRESION_SISTOLICA            FLOAT,
  GLUCOSA_MG_DL                FLOAT,
  PARIDAD                      NUMBER(2,0),
  CA125_U_ML                   FLOAT,
  RESULTADO_IMAGEN             VARCHAR(20),
  FUENTE                       VARCHAR(30),
  PERIODO                      VARCHAR(10),
  RIESGO_ONCOLOGICO            NUMBER(1,0)
) COMMENT = 'Lote SINTÉTICO de prueba cargado por Stage; no son pacientes reales';

-- Trazabilidad: origen y momento de cada carga (arquitectura VII-A.2)
CREATE TABLE IF NOT EXISTS AUDITORIA.REGISTRO_CARGAS (
  FECHA_REGISTRO    TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
  TABLA_DESTINO     VARCHAR,
  ARCHIVO_ORIGEN    VARCHAR,
  FILAS_LEIDAS      NUMBER,
  FILAS_CARGADAS    NUMBER,
  ESTADO            VARCHAR,
  PRIMER_ERROR      VARCHAR,
  FECHA_CARGA       TIMESTAMP_LTZ
);

-- Bitácora: fecha, versión, regla, campo y registros afectados (VI-D.3)
CREATE TABLE IF NOT EXISTS AUDITORIA.BITACORA_TRANSFORMACIONES (
  FECHA_REGISTRO      TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
  VERSION             VARCHAR,
  ETAPA               VARCHAR,
  REGLA               VARCHAR,
  CAMPO               VARCHAR,
  REGISTROS_AFECTADOS NUMBER
);

-- ---------------------------------------------------------------------
-- Control de acceso (opcional): rol de solo lectura para integrantes que
-- consultan los resultados sin modificar datos. Descomente para usarlo.
-- ---------------------------------------------------------------------
-- CREATE ROLE IF NOT EXISTS ONCORISK_LECTOR
-- GRANT USAGE ON WAREHOUSE ONCORISK_WH TO ROLE ONCORISK_LECTOR
-- GRANT USAGE ON DATABASE ONCORISK_DB TO ROLE ONCORISK_LECTOR
-- GRANT USAGE ON SCHEMA ONCORISK_DB.ANALYTICAL TO ROLE ONCORISK_LECTOR
-- GRANT SELECT ON ALL TABLES IN SCHEMA ONCORISK_DB.ANALYTICAL TO ROLE ONCORISK_LECTOR
-- GRANT SELECT ON FUTURE TABLES IN SCHEMA ONCORISK_DB.ANALYTICAL TO ROLE ONCORISK_LECTOR
