# Resultados y limitaciones de la ejecución sintética

Versión 1.1, 17 de noviembre de 2026. Fuente: `reports/resumen_pipeline.json` y el panel **Evaluación del modelo**. La muestra fue generada artificialmente y no representa pacientes del hospital.

## Datos

20.000 pacientes sintéticos, uno por fila, sin faltantes, duplicados ni valores imposibles (`reports/calidad_raw.json`, `reports/calidad_clean.json`). Prevalencia de la etiqueta 1: 24,9 %. Partición agrupada por paciente: 16.000 de desarrollo y 4.000 de prueba reservada.

## Selección del modelo (solo desarrollo)

Validación cruzada de 5 pliegues, estratificada y agrupada por paciente, con AUROC como métrica principal:

| Modelo | AUROC medio (DE) | Brier fuera de pliegue |
|---|---:|---:|
| **Regresión logística** | **0,7253 (0,0129)** | **0,1620** |
| Random Forest | 0,7221 (0,0125) | 0,1633 |
| XGBoost | 0,7139 (0,0137) | 0,1646 |
| Modelo nulo (prevalencia) | 0,5000 | 0,1872 |

Se seleccionó la regresión logística: tiene el AUROC más alto, la mejor calibración y es la más interpretable. Las diferencias con Random Forest son pequeñas. El umbral técnico, elegido maximizando F1 en desarrollo, es 0,25 (`documentacion/decisiones_modelado.md`).

## Evaluación en la prueba reservada (4.000 pacientes)

| Métrica | Estimación | IC 95 % bootstrap | Modelo nulo |
|---|---:|---|---:|
| AUROC | 0,734 | 0,714 – 0,751 | 0,500 |
| PR-AUC (precisión promedio) | 0,497 | 0,469 – 0,525 | 0,250 |
| Brier | **0,160** | 0,156 – 0,164 | 0,187 |
| Sensibilidad | 0,641 | 0,612 – 0,672 | 0 |
| Especificidad | 0,703 | 0,688 – 0,718 | 1 |
| Valor predictivo positivo | 0,418 | 0,400 – 0,435 | — |
| F1 | 0,506 | 0,485 – 0,526 | 0 |

Matriz de confusión con umbral 0,25: 640 verdaderos positivos, 893 falsos positivos, 2.109 verdaderos negativos y 358 falsos negativos. Bootstrap percentil estratificado de 500 réplicas, condicionado al modelo ya ajustado.

**Calibración.** La curva de calibración sigue la diagonal: por ejemplo, en el decil más alto la probabilidad media estimada es 0,61 y la proporción observada 0,615.

## Contraste de la hipótesis (sección V-D)

| Condición | Resultado | ¿Cumple? |
|---|---|---|
| AUROC > 0,50 | 0,734 (IC 0,714 – 0,751) | Sí |
| Brier < Brier del modelo nulo | 0,160 frente a 0,187 (el IC completo queda por debajo) | Sí |
| Identificar variables con contribución predictiva | Hallazgo de imagen, antecedente familiar, tabaquismo, categoría de IMC y edad | Sí |

Se rechaza la hipótesis nula **en esta muestra sintética**. Como la etiqueta no tiene fecha índice ni seguimiento, la salida es una probabilidad de clasificación de la condición registrada, no un riesgo futuro.

## Variables con mayor contribución

Coeficientes del modelo (log-odds sobre variables escaladas o codificadas): hallazgo de imagen sospechoso (+0,99) frente a normal (−0,99), antecedente familiar de cáncer (+0,35 / −0,46), tabaquismo activo (+0,26) frente a nunca (−0,29), bajo peso (−0,26) y edad (+0,24 por desviación estándar). Describen el modelo, no causas: el generador sintético impuso esas relaciones.

## Limitaciones

- Datos sintéticos: las asociaciones las definió el generador; no son evidencia clínica ni causal.
- Validación solo interna, sin validación externa, temporal ni clínica.
- La prueba reservada se evaluó dos veces (antes y después de retirar los pesos de clase). La decisión se tomó con datos de desarrollo, pero esto debe declararse (`decisiones_modelado.md`, punto 1).
- El umbral de 0,25 es técnico; con él, casi 6 de cada 10 alertas son falsos positivos (VPP 0,42).
- PR-AUC se calcula como precisión promedio (`average_precision_score`) y puede diferir del área trapezoidal.
- La infraestructura de Snowflake (`sql/00` a `sql/03`, incluida la carga por Stage) se verificó en la cuenta del proyecto con resultado idéntico al de Python. El recorrido completo `--backend snowflake --transformar-con-sql` (lectura del lote en Snowflake, CLEAN y ANALYTICAL con SQL dentro de Snowflake y entrenamiento) produjo el mismo modelo, umbral y métricas de prueba que la corrida local; solo Random Forest y XGBoost variaron en el cuarto decimal del AUROC de validación cruzada por el orden de las filas.
- No utilizar el modelo para diagnóstico, tratamiento ni decisiones clínicas.
