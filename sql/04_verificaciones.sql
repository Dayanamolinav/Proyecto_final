-- =====================================================================
-- OncoRisk · 04 · Verificaciones (solo lectura)
-- Uso: python scripts/ejecutar_sql.py sql/04_verificaciones.sql
-- Sirve como evidencia para la matriz de pruebas (PF-20 y PF-21).
-- =====================================================================

USE DATABASE ONCORISK_DB;

-- Última carga por Stage registrada
SELECT FECHA_REGISTRO, TABLA_DESTINO, ARCHIVO_ORIGEN, FILAS_LEIDAS, FILAS_CARGADAS, ESTADO, PRIMER_ERROR
FROM AUDITORIA.REGISTRO_CARGAS
ORDER BY FECHA_REGISTRO DESC
LIMIT 3;

-- Filas por capa
SELECT
  (SELECT COUNT(*) FROM RAW.PACIENTES_PRUEBA_SINTETICOS) AS ATERRIZAJE,
  (SELECT COUNT(*) FROM RAW.CLINICAL_RAW)                AS RAW_CAPA,
  (SELECT COUNT(*) FROM CLEAN.CLINICAL_CLEAN)            AS CLEAN_CAPA,
  (SELECT COUNT(*) FROM ANALYTICAL.CLINICAL_ANALYTICAL)  AS ANALYTICAL_CAPA;

-- El lote recargado por Stage debe coincidir con la capa RAW (mismo CSV)
SELECT
  (SELECT COUNT(*) FROM (SELECT * FROM RAW.PACIENTES_PRUEBA_SINTETICOS MINUS SELECT * FROM RAW.CLINICAL_RAW) A) AS SOLO_EN_ATERRIZAJE,
  (SELECT COUNT(*) FROM (SELECT * FROM RAW.CLINICAL_RAW MINUS SELECT * FROM RAW.PACIENTES_PRUEBA_SINTETICOS) B) AS SOLO_EN_RAW;

-- Pasos registrados en la bitácora de transformaciones
SELECT ETAPA, REGLA, REGISTROS_AFECTADOS
FROM AUDITORIA.BITACORA_TRANSFORMACIONES
ORDER BY FECHA_REGISTRO;
