# Diccionario de datos de la muestra sintética

Versión 1.0; fecha 2026-11-13. Responsable: equipo del proyecto, pendiente de asignar. Este diccionario describe el generador actual y no predice la estructura del CSV del hospital. Los campos y reglas reales deben revisarse antes de incorporarlos.

| Variable | Tipo / valores simulados | Función y tratamiento |
|---|---|---|
| `id_paciente` | texto `PAC-xxxxx` | Identificador ficticio para trazabilidad y partición por paciente; no predictor. |
| `nombre`, `documento_identidad` | texto | Identificadores directos ficticios de prueba; no predictores. El mecanismo de protección para datos reales debe diseñarse y aprobarse antes de cargar archivos. |
| `edad` | numérica, años | Predictor. Limpieza actual: fuera de 0–120 se excluye; rango de entrada de la app y uso real requieren revisión. |
| `sexo` | categoría F/M | Predictor simulado; categorías no son una definición clínica universal. |
| `antecedente_familiar_cancer` | binaria 0/1 | Predictor simulado. |
| `antecedente_personal_benigno` | binaria 0/1 | Predictor simulado. |
| `tabaquismo` | nunca, exfumador, activo | Predictor simulado. |
| `imc` | numérica kg/m² | Predictor, faltantes simulados. Limpieza actual excluye fuera de 10–80. |
| `presion_sistolica` | numérica mmHg | Predictor con faltantes simulados. |
| `glucosa_mg_dl` | numérica mg/dL | Predictor con faltantes simulados. |
| `paridad` | entero | Predictor simulado; pertinencia real pendiente. |
| `ca125_u_ml` | numérica U/mL | Biomarcador ficticio; pertinencia, momento de medición y riesgo de fuga deben evaluarse en la fuente real. |
| `resultado_imagen` | normal, indeterminado, sospechoso | Predictor simulado con faltantes; verificar momento de adquisición antes de usar datos reales. |
| `fuente` | consulta_externa, urgencias, remision | Procedencia ficticia; metadato, no predictor. |
| `periodo` | 2023-S1, 2023-S2, 2024-S1, 2024-S2 | Período simulado; metadato, no predictor. |
| `riesgo_oncologico` | binaria 0/1 | Etiqueta sintética de clasificación, no estimación validada de riesgo futuro. |
| `fecha_diagnostico_confirmado`, `tratamiento_indicado`, `comentario_oncologo` | fecha/texto | Campos posteriores al desenlace simulados: se eliminan antes del modelado para evitar fuga. |
| `imc_categoria` | bajo_peso, normal, sobrepeso, obesidad | Derivada de IMC con los cortes definidos en el código; predicción sintética. |
| `grupo_etario` | <40, 40-54, 55-69, 70+ | Derivada de edad con los cortes definidos en el código; revisar límites y nombres si se cambia la definición. |

El preprocesamiento emplea imputación numérica por mediana, imputación categórica por moda, escalado y codificación categórica; se ajusta en entrenamiento. La lista fija de campos posteriores no garantiza eliminar todas las fugas de un archivo desconocido. La unidad de análisis real y la disponibilidad temporal de cada predictor deben confirmarse.
