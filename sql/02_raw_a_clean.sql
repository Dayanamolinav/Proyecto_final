-- =====================================================================
-- OncoRisk · 02 · RAW -> CLEAN (metodología Fase 3, arquitectura VII-B.4)
--
-- Entrada:  RAW.CLINICAL_RAW (lote recibido, ya sin nombre ni documento)
-- Salida:   CLEAN.CLINICAL_CLEAN
-- Reglas (mismas que src/processing/cleaning.py, verificadas por
-- tests/test_sql_transformations.py):
--   1. Eliminar duplicados exactos.
--   2. Eliminar valores clínicamente imposibles: edad fuera de [0, 120] o
--      IMC fuera de [10, 80] (los nulos se conservan para imputarlos luego).
--   3. Eliminar columnas posteriores al desenlace (fuga de información):
--      fecha_diagnostico_confirmado, tratamiento_indicado, comentario_oncologo.
--      Se logra listando solo las columnas permitidas en el SELECT.
--   4. Eliminar registros sin variable objetivo.
-- La imputación, codificación y escalado NO se hacen aquí: dependen de los
-- datos y deben ajustarse solo con el conjunto de desarrollo (scikit-learn).
-- =====================================================================

-- Los duplicados se comparan con TODAS las columnas recibidas (subconsulta D),
-- antes de descartar las columnas con fuga, igual que en Python.
CREATE OR REPLACE TABLE CLEAN.CLINICAL_CLEAN AS
SELECT
  D.ID_PACIENTE, D.EDAD, D.SEXO, D.ANTECEDENTE_FAMILIAR_CANCER, D.ANTECEDENTE_PERSONAL_BENIGNO,
  D.TABAQUISMO, D.IMC, D.PRESION_SISTOLICA, D.GLUCOSA_MG_DL, D.PARIDAD, D.CA125_U_ML,
  D.RESULTADO_IMAGEN, D.FUENTE, D.PERIODO, D.RIESGO_ONCOLOGICO
FROM (SELECT DISTINCT * FROM RAW.CLINICAL_RAW) D
WHERE (D.EDAD IS NULL OR D.EDAD BETWEEN 0 AND 120)
  AND (D.IMC IS NULL OR D.IMC BETWEEN 10 AND 80)
  AND D.RIESGO_ONCOLOGICO IS NOT NULL;

-- Bitácora de transformaciones
INSERT INTO AUDITORIA.BITACORA_TRANSFORMACIONES (VERSION, ETAPA, REGLA, CAMPO, REGISTROS_AFECTADOS)
SELECT 'sql-v1', 'CLEAN', 'eliminacion_duplicados_exactos', '*',
       (SELECT COUNT(*) FROM RAW.CLINICAL_RAW)
       - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM RAW.CLINICAL_RAW) D);

INSERT INTO AUDITORIA.BITACORA_TRANSFORMACIONES (VERSION, ETAPA, REGLA, CAMPO, REGISTROS_AFECTADOS)
SELECT 'sql-v1', 'CLEAN', 'eliminacion_valores_clinicamente_imposibles', 'EDAD [0,120], IMC [10,80]',
       COUNT(*)
FROM (SELECT DISTINCT * FROM RAW.CLINICAL_RAW) D
WHERE NOT ((D.EDAD IS NULL OR D.EDAD BETWEEN 0 AND 120) AND (D.IMC IS NULL OR D.IMC BETWEEN 10 AND 80));

INSERT INTO AUDITORIA.BITACORA_TRANSFORMACIONES (VERSION, ETAPA, REGLA, CAMPO, REGISTROS_AFECTADOS)
SELECT 'sql-v1', 'CLEAN', 'eliminacion_registros_sin_variable_objetivo', 'RIESGO_ONCOLOGICO',
       COUNT(*)
FROM (SELECT DISTINCT * FROM RAW.CLINICAL_RAW) D
WHERE (D.EDAD IS NULL OR D.EDAD BETWEEN 0 AND 120)
  AND (D.IMC IS NULL OR D.IMC BETWEEN 10 AND 80)
  AND D.RIESGO_ONCOLOGICO IS NULL;

INSERT INTO AUDITORIA.BITACORA_TRANSFORMACIONES (VERSION, ETAPA, REGLA, CAMPO, REGISTROS_AFECTADOS)
SELECT 'sql-v1', 'CLEAN', 'construccion_tabla_clean', 'CLEAN.CLINICAL_CLEAN', COUNT(*)
FROM CLEAN.CLINICAL_CLEAN;
