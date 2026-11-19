# Decisiones de modelado y de procesamiento

Versión 1.1, 17 de noviembre de 2026. Datos: lote sintético de 20.000 pacientes (`data/synthetic/pacientes_sinteticos_20000.csv`, el mismo cargado en `RAW.PACIENTES_PRUEBA_SINTETICOS`). Evidencia: `reports/resumen_pipeline.json` y `reports/registro_experimentos.jsonl`.

Este documento responde a lo que el Informe 2 dejó para decidir con los datos de desarrollo: umbral, tratamiento del desbalance y de los valores extremos, y la herramienta de las transformaciones.

## 1. Pesos de clase: se retiran

**Antes.** Los tres modelos se entrenaban con pesos de clase inversamente proporcionales a la frecuencia (la clase 1 pesaba 3 veces más).

**Problema.** Los pesos desplazan todas las probabilidades hacia arriba. El modelo seguía ordenando bien a los pacientes (AUROC), pero sus probabilidades dejaban de corresponder a las frecuencias observadas, y el Brier quedaba peor que el del modelo nulo. Eso incumplía la segunda condición de la hipótesis de investigación (sección V-D.1).

**Evidencia usada para decidir (solo desarrollo, 16.000 pacientes, validación cruzada de 5 pliegues agrupada por paciente, predicciones fuera de pliegue):**

| Modelo | AUROC con pesos | AUROC sin pesos | Brier con pesos | Brier sin pesos | Brier nulo |
|---|---:|---:|---:|---:|---:|
| Regresión logística | 0,7249 | 0,7249 | 0,2087 | **0,1620** | 0,1872 |
| Random Forest | 0,7210 | 0,7213 | 0,2099 | **0,1633** | 0,1872 |
| XGBoost | 0,7116 | 0,7134 | 0,2060 | **0,1646** | 0,1872 |

Retirar los pesos no cambia la discriminación y lleva el Brier por debajo del nulo en los tres modelos. El desbalance se trata con el umbral (punto 2), que cambia la clasificación sin deformar las probabilidades. La opción sigue disponible en `config.USE_CLASS_WEIGHTS`.

**Limitación que debe declararse.** El problema se detectó al leer la evaluación de la prueba reservada de la ejecución anterior (Brier 0,2064 frente a 0,1872 del nulo). La corrección se justificó y se eligió exclusivamente con datos de desarrollo, como muestra la tabla, pero la prueba reservada se evaluó dos veces: una con pesos y otra sin ellos. Por tanto, la evaluación final ya no es estrictamente "una sola vez" como pedía la Fase 4. Como el cambio no se ajustó a la prueba (no se probaron variantes contra ella) y los resultados de desarrollo y de prueba coinciden (Brier 0,162 y 0,160), el sesgo esperado es pequeño, pero debe mencionarse en el Informe 3.

Resultados de la prueba en ambas ejecuciones (4.000 pacientes, mismo modelo seleccionado):

| Ejecución | AUROC | Brier | Brier nulo | Sensibilidad | Especificidad | VPP | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anterior (con pesos, umbral 0,50) | 0,7345 | 0,2064 | 0,1872 | 0,643 | 0,703 | 0,419 | 0,507 |
| Actual (sin pesos, umbral 0,25) | 0,7344 | **0,1600** | 0,1872 | 0,641 | 0,703 | 0,418 | 0,506 |

La clasificación prácticamente no cambia (umbral 0,50 con pesos ≈ umbral 0,25 sin pesos); lo que mejora es la calibración: ahora una probabilidad de 61 % corresponde a un 61,5 % observado, en lugar de 82 % frente a 61,5 %.

## 2. Umbral de clasificación: elegido en desarrollo

- **Regla:** se maximiza F1 sobre las predicciones fuera de pliegue del modelo seleccionado, en una rejilla de 0,05 a 0,95 con paso 0,01. En empate se elige el umbral más bajo, que favorece la sensibilidad.
- **Resultado:** umbral **0,25** (F1 de desarrollo 0,501).
- Se guarda junto con el modelo (`threshold` en `models/modelo_seleccionado.joblib`) y se aplica sin cambios en la prueba reservada.
- Es una decisión técnica del prototipo, no un punto de corte clínico. El formulario muestra la probabilidad, no una clasificación del paciente.
- El control deslizante del panel estadístico sigue disponible para explorar otros umbrales, pero esos valores no se usan para elegir nada.

## 3. Valores extremos

| Tipo | Regla | Tratamiento |
|---|---|---|
| Imposibles | Edad fuera de [0, 120], IMC fuera de [10, 80] | Se eliminan en CLEAN (`sql/02_raw_a_clean.sql` y `cleaning.py`) y se registran en la bitácora |
| Atípicos plausibles | Fuera de Q1 − 1,5·IQR o Q3 + 1,5·IQR | Se conservan y se reportan en `reports/calidad_*.json`, en el panel y en el cuaderno |

No se recortan ni se eliminan los atípicos plausibles porque, en datos clínicos, suelen ser justamente los casos de interés (por ejemplo, CA-125 elevado). Su influencia se limita con la imputación por mediana, el escalado estándar ajustado solo en desarrollo y modelos de árboles, insensibles a la escala. En el lote de 20.000 no hay valores imposibles.

## 4. Transformaciones en SQL y en Python

El informe prevé las transformaciones "principalmente mediante SQL" en Snowflake. Ahora existen así:

| Paso | SQL (Snowflake) | Python equivalente (local) |
|---|---|---|
| Objetos, capas, Stage | `sql/00_setup_inicial.sql` | — |
| Ingesta por Stage | `sql/01_ingesta_stage.sql` + `scripts/cargar_csv_snowflake.py` | `load_local_synthetic_csv()` |
| RAW → CLEAN | `sql/02_raw_a_clean.sql` | `src/processing/cleaning.py` |
| CLEAN → ANALYTICAL | `sql/03_clean_a_analytical.sql` | `src/processing/feature_engineering.py` |
| Bitácora | `AUDITORIA.BITACORA_TRANSFORMACIONES` | `logs/bitacora_trazabilidad.jsonl` |

- `python scripts/run_pipeline.py --backend snowflake --transformar-con-sql` construye CLEAN y ANALYTICAL dentro de Snowflake.
- `tests/test_sql_transformations.py` ejecuta los mismos archivos SQL en SQLite y comprueba que producen exactamente las mismas filas que Python, tanto en un lote con duplicados, faltantes, valores imposibles y columnas con fuga como en los 20.000.
- La imputación, la codificación y el escalado **no** se hacen en SQL: dependen de los datos y deben ajustarse solo con el conjunto de desarrollo dentro de cada pliegue. Por eso viven en el `Pipeline` de scikit-learn.
- **Verificado en Snowflake el 17 de noviembre de 2026** (Snowsight, warehouse ONCORISK_WH): `sql/00` creó el esquema AUDITORIA, el formato de archivo y el Stage; `sql/02` y `sql/03` reconstruyeron CLEAN y ANALYTICAL desde `RAW.CLINICAL_RAW`. La comparación `MINUS` en ambos sentidos contra la tabla ANALYTICAL que había escrito Python dio 0 diferencias (20.000 filas, 0 obligatorios nulos, 0 pacientes repetidos), y la bitácora registró los 5 pasos. La carga por Stage (`sql/01`, con `scripts/cargar_csv_snowflake.py`) también se verificó: PUT del CSV, COPY INTO a una tabla temporal (20.000 leídas, 20.000 cargadas, estado LOADED, sin errores) e INSERT OVERWRITE en la tabla de aterrizaje, que quedó idéntica a `RAW.CLINICAL_RAW`. Evidencia reproducible con `sql/04_verificaciones.sql`. Finalmente, `python scripts/run_pipeline.py --backend snowflake --transformar-con-sql` se ejecutó de extremo a extremo con el mismo modelo, umbral (0,25) y métricas de prueba que la corrida local.

## 5. Variables derivadas con valores faltantes

Según la versión de pandas, un IMC o una edad faltante producía la categoría de texto `"nan"` (pandas 2) o un valor vacío (pandas 3). Ahora siempre queda vacío y se imputa con la moda del conjunto de desarrollo, igual en Python y en SQL. En el lote de 20.000 no hay faltantes, así que no cambia los resultados actuales.
