-- =====================================================================
-- OncoRisk · 03 · CLEAN -> ANALYTICAL (metodología Fase 3)
--
-- Entrada:  CLEAN.CLINICAL_CLEAN
-- Salida:   ANALYTICAL.CLINICAL_ANALYTICAL (tabla que lee el entrenamiento)
-- Construye las variables derivadas con los mismos cortes que
-- src/processing/derived_variables.py (intervalos cerrados a la izquierda):
--   GRUPO_ETARIO:  <40 | 40-54 | 55-69 | 70+
--   IMC_CATEGORIA: bajo_peso (<18,5) | normal (<25) | sobrepeso (<30) | obesidad
-- Un valor faltante de edad o IMC deja la variable derivada en NULL, igual
-- que en Python; luego se imputa con la moda del conjunto de desarrollo.
-- =====================================================================

CREATE OR REPLACE TABLE ANALYTICAL.CLINICAL_ANALYTICAL AS
SELECT
  ID_PACIENTE, FUENTE, PERIODO,
  EDAD, IMC, PRESION_SISTOLICA, GLUCOSA_MG_DL, PARIDAD, CA125_U_ML,
  SEXO, ANTECEDENTE_FAMILIAR_CANCER, ANTECEDENTE_PERSONAL_BENIGNO, TABAQUISMO, RESULTADO_IMAGEN,
  CASE
    WHEN IMC >= 0    AND IMC < 18.5 THEN 'bajo_peso'
    WHEN IMC >= 18.5 AND IMC < 25   THEN 'normal'
    WHEN IMC >= 25   AND IMC < 30   THEN 'sobrepeso'
    WHEN IMC >= 30   AND IMC < 101  THEN 'obesidad'
    ELSE NULL
  END AS IMC_CATEGORIA,
  CASE
    WHEN EDAD >= 0  AND EDAD < 40  THEN '<40'
    WHEN EDAD >= 40 AND EDAD < 55  THEN '40-54'
    WHEN EDAD >= 55 AND EDAD < 70  THEN '55-69'
    WHEN EDAD >= 70 AND EDAD < 121 THEN '70+'
    ELSE NULL
  END AS GRUPO_ETARIO,
  RIESGO_ONCOLOGICO
FROM CLEAN.CLINICAL_CLEAN;

INSERT INTO AUDITORIA.BITACORA_TRANSFORMACIONES (VERSION, ETAPA, REGLA, CAMPO, REGISTROS_AFECTADOS)
SELECT 'sql-v1', 'ANALYTICAL', 'construccion_variables_derivadas', 'IMC_CATEGORIA, GRUPO_ETARIO', COUNT(*)
FROM ANALYTICAL.CLINICAL_ANALYTICAL;

-- Criterios de aceptación de la tabla analítica (VI-D.3): campos obligatorios
-- válidos y sin pacientes duplicados. Ambas cifras deben ser 0.
SELECT
  SUM(CASE WHEN ID_PACIENTE IS NULL OR RIESGO_ONCOLOGICO IS NULL THEN 1 ELSE 0 END) AS OBLIGATORIOS_NULOS,
  COUNT(*) - COUNT(DISTINCT ID_PACIENTE) AS PACIENTES_REPETIDOS
FROM ANALYTICAL.CLINICAL_ANALYTICAL;
