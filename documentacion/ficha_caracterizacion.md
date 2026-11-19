# Ficha de caracterización: datos sintéticos

Versión: 1.0. Fecha: 2026-11-13. Responsable: equipo del proyecto, pendiente de asignar.

Esta ficha describe exclusivamente la ejecución previa del generador sintético, no la población de un hospital. Fuente: `src/data_layer/synthetic_data.py`; configuración predeterminada: semilla 42 y 1.200 pacientes generados, con duplicados y valores inválidos introducidos deliberadamente. La etiqueta 2023-S1 a 2024-S2 es simulada; no representa un período de atención real. Institución, autorización de uso y tipo específico de cáncer: no aplican a esta muestra sintética; pendientes de confirmar para datos reales.

| Indicador de la ejecución previa | Resultado |
|---|---:|
| Registros en RAW | 1.212 |
| Duplicados exactos detectados en RAW | 12 |
| Edad fuera del intervalo 0–120 en RAW | 2 |
| Registros en CLEAN y ANALYTICAL | 1.198 |
| Identificadores repetidos en CLEAN | 0 |
| Clase 0 en CLEAN | 908 (75,79 %) |
| Clase 1 en CLEAN | 290 (24,21 %) |

La cantidad objetivo es una etiqueta simulada de clasificación binaria; no hay fecha índice, horizonte de seguimiento ni evento clínico futuro validado. La probabilidad del modelo solo describe la condición simulada. La creación sintética incorpora asociaciones definidas artificialmente, de modo que no puede inferirse una asociación clínica ni causal. La ficha debe actualizarse para cada nueva ejecución a partir de `reports/calidad_raw.json`, `reports/calidad_clean.json` y `reports/resumen_pipeline.json`; las cifras anteriores son de la ejecución previa a las correcciones.

En CLEAN se observaron faltantes en CA-125 (17,20 %), imagen (10,85 %), glucosa (10,18 %), presión (8,60 %) e IMC (4,09 %). La revisión IQR señala valores extremos potenciales, no errores demostrados. La imputación se ajusta dentro de la canalización de entrenamiento. El tratamiento en datos reales necesitará validación específica.
