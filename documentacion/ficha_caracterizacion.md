# Ficha de caracterización: datos sintéticos

Versión: 1.1. Fecha: 2026-11-17. Responsable: equipo del proyecto.

Esta ficha describe el lote sintético usado para entrenar el modelo actual, no la población de un hospital.

| Campo (sección VI-B del informe) | Valor |
|---|---|
| Institución o fuente | Generador sintético del prototipo (`src/data_layer/synthetic_data.py`, semilla 42) |
| Archivo | `data/synthetic/pacientes_sinteticos_20000.csv`; SHA-256 `bf2078859176c2292618f8688944f8d14b2ab4688754c3bdd5ac87e4b2bc652a` |
| Carga | Stage de Snowflake a `RAW.PACIENTES_PRUEBA_SINTETICOS` (`sql/01_ingesta_stage.sql`); copia local idéntica para trabajar sin conexión |
| Período cubierto | Semestres simulados 2023-S1 a 2024-S2 (procedencia, no fechas de seguimiento) |
| Tipo de enfermedad oncológica | No aplica: etiqueta simulada `riesgo_oncologico` |
| Formato | CSV UTF-8, 15 columnas, una fila por paciente |
| Registros iniciales | 20.000 |
| Pacientes únicos | 20.000 |
| Variables | 11 predictores, 2 metadatos (`fuente`, `periodo`), identificador y etiqueta (ver `diccionario_sintetico.md`) |
| Proporción de clases | Clase 0: 15.012 (75,06 %); clase 1: 4.988 (24,94 %) |
| Anonimización | Sin nombre ni documento en el archivo; identificador ficticio `PAC-xxxxx` |
| Autorización de uso | No aplica (datos generados); pendiente para datos reales |

## Flujo de selección

| Etapa | Registros |
|---|---:|
| RAW | 20.000 |
| Duplicados exactos eliminados | 0 |
| Valores clínicamente imposibles eliminados (edad, IMC) | 0 |
| Registros sin variable objetivo eliminados | 0 |
| CLEAN y ANALYTICAL | 20.000 |
| Desarrollo / prueba reservada (agrupada por paciente) | 16.000 / 4.000 |

No hay valores faltantes. La revisión IQR señala atípicos plausibles que se conservan (`decisiones_modelado.md`, punto 3). Hay 3.567 registros con sexo M y paridad mayor que 0, producto de la simulación independiente de campos; se conservan y se señalan en el panel de calidad.

La etiqueta es una clasificación binaria simulada, sin fecha índice, horizonte de seguimiento ni evento clínico futuro validado: la probabilidad del modelo describe solo la condición simulada. Esta ficha debe actualizarse en cada ejecución a partir de `reports/calidad_raw.json`, `reports/calidad_clean.json` y `reports/resumen_pipeline.json`.

**Ejecución anterior (referencia).** La versión 1.0 de esta ficha describía un lote generado de 1.212 registros (12 duplicados y 2 edades inválidas eliminados, 1.198 en ANALYTICAL). Ese lote ya no se usa para entrenar; el generador se conserva para las pruebas de limpieza.
