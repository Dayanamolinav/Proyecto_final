# Prototipo inicial — Identificación de características asociadas al riesgo oncológico

Backend y prototipo de prueba de concepto para el proyecto *"Diseño y desarrollo de una
solución de ingeniería de datos para la identificación de características asociadas al
riesgo oncológico mediante inteligencia artificial"* (Universidad del Norte).

> **Importante:** el conjunto de datos real del hospital universitario aún no ha sido
> entregado. Este prototipo usa un **generador de datos clínicos sintéticos**
> (`src/data_layer/synthetic_data.py`) con una estructura plausible (edad, sexo,
> antecedentes, hábitos, biomarcador, hallazgo de imagen, etc.) para poder construir y
> validar de punta a punta toda la canalización descrita en el informe. **Nada de esto
> son datos de pacientes reales.** Cuando el archivo CSV real esté disponible, basta con
> ejecutar `python scripts/run_pipeline.py --csv ruta/al/archivo.csv`: el resto del
> pipeline no cambia.

## Cómo se relaciona con el informe

El código sigue la arquitectura lógica y física descrita en las secciones VI y VII del
informe (Fig. 3 y Fig. 4), simulando localmente lo que en producción sería Snowflake:

| Etapa del informe                          | Módulo en este repo                              |
|---------------------------------------------|---------------------------------------------------|
| Fuente de datos / Ingesta (Stage, CSV)       | `src/data_layer/ingestion.py`                     |
| Almacenamiento RAW / CLEAN / ANALYTICAL      | `src/data_layer/storage.py` (+ `data/raw`, `data/clean`, `data/analytical`) |
| Calidad de datos (completitud, duplicados, validez, outliers, balance) | `src/processing/quality.py` |
| Procesamiento (limpieza, sin fuga de información) | `src/processing/cleaning.py`                |
| Modelado (variables derivadas, codificación) | `src/processing/feature_engineering.py`           |
| Análisis y modelado (regresión logística, Random Forest, XGBoost) | `src/modeling/train.py` |
| Evaluación (AUROC, PR-AUC, sensibilidad, especificidad, F1, Brier, calibración) | `src/modeling/evaluate.py` |
| Interpretación de variables (contribución predictiva, no causal) | `src/modeling/explain.py` |
| Registro de experimentos (reproducibilidad) | `src/modeling/registry.py`                        |
| Gobernanza, seguridad y trazabilidad         | `src/data_layer/traceability.py` (bitácora en `logs/`) |
| Visualización y entrega (aplicación web)     | `app/streamlit_app.py`                            |

El pipeline completo (`src/pipeline.py`) orquesta estas etapas en el mismo orden del
ciclo CRISP-DM adaptado del proyecto (Fig. 2, Anexo A): comprensión del problema →
comprensión y preparación de datos → modelado → evaluación → despliegue académico
(prototipo).

## Estructura del proyecto

```
oncorisk_prototype/
├── config.py                      # rutas, semilla, nombres de columnas, umbrales
├── requirements.txt
├── src/
│   ├── data_layer/
│   │   ├── synthetic_data.py      # generador de datos de PRUEBA (no reales)
│   │   ├── ingestion.py           # ingesta + persistencia en RAW (seudonimizada)
│   │   ├── storage.py             # capas RAW / CLEAN / ANALYTICAL (parquet local)
│   │   └── traceability.py        # bitácora de trazabilidad (logs/*.jsonl)
│   ├── processing/
│   │   ├── quality.py             # reporte de calidad de datos
│   │   ├── cleaning.py            # RAW -> CLEAN (duplicados, validez, fuga de info)
│   │   └── feature_engineering.py # CLEAN -> ANALYTICAL (variables derivadas, encoding)
│   ├── modeling/
│   │   ├── train.py               # partición, CV, pesos de clase, 3 algoritmos
│   │   ├── evaluate.py            # métricas + modelo nulo + hipótesis de investigación
│   │   ├── explain.py             # interpretabilidad (coeficientes / importancias)
│   │   └── registry.py            # registro de experimentos (reports/*.jsonl)
│   └── pipeline.py                # orquestador de punta a punta
├── app/
│   └── streamlit_app.py           # prototipo web (entrada de variables + estimación)
├── scripts/
│   └── run_pipeline.py            # CLI: python scripts/run_pipeline.py [--csv ...]
├── data/{raw,clean,analytical}/   # capas de datos (parquet)
├── models/                        # modelo seleccionado (joblib)
├── reports/                       # calidad, registro de experimentos, resumen
└── logs/                          # bitácora de trazabilidad
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt

# 1) Ejecutar el pipeline completo (genera datos sintéticos, construye RAW/CLEAN/
#    ANALYTICAL, entrena y compara los 3 modelos, evalúa y persiste el seleccionado)
python scripts/run_pipeline.py

# 2) Levantar el prototipo web
streamlit run app/streamlit_app.py
```

Para usar el archivo real del hospital cuando esté disponible:

```bash
python scripts/run_pipeline.py --csv /ruta/al/archivo_hospital.csv
```

## Decisiones de diseño relevantes

- **Sin fuga de información:** `cleaning.drop_leaky_columns` elimina explícitamente
  cualquier variable generada después del desenlace (p. ej. tratamiento indicado),
  siguiendo la definición de "Fuga de información" del Marco Conceptual (Sección V-C).
- **Seudonimización antes de RAW:** `storage.pseudonymize_direct_identifiers` sustituye
  identificadores directos (nombre, documento) por un hash antes de escribir en RAW,
  conforme a la Sección E (Consideraciones éticas).
- **Partición y validación:** partición entrenamiento/prueba estratificada (80/20) +
  validación cruzada de 5 pliegues sobre el conjunto de desarrollo; el conjunto de
  prueba se evalúa **una sola vez**, después de seleccionar el modelo (Fase 4).
- **Manejo de desbalance:** pesos de clase inversamente proporcionales a la frecuencia
  (mismo criterio usado en el notebook de referencia PCA/LDA que compartiste), aplicados
  a regresión logística y Random Forest, y `scale_pos_weight` en XGBoost.
- **Métricas:** AUROC, PR-AUC, sensibilidad, especificidad, VPP, F1, Brier score y curva
  de calibración, más contraste explícito contra un **modelo nulo** (prevalencia
  observada) para evaluar la Hipótesis de investigación (Sección V-D).
- **Interpretabilidad:** coeficientes (regresión logística) o importancias (modelos de
  árboles) se reportan como *contribución predictiva*, nunca como causalidad.
- **Trazabilidad:** cada carga y transformación queda registrada en
  `logs/bitacora_trazabilidad.jsonl`; cada entrenamiento queda registrado en
  `reports/registro_experimentos.jsonl`, con semilla, hiperparámetros, variables y
  métricas, para reproducibilidad.
- **No diagnóstico:** la app Streamlit incluye advertencias explícitas de que es un
  prototipo académico/experimental que no reemplaza la valoración profesional,
  siguiendo la Sección VI-E del informe.

## Próximos pasos naturales (cuando lleguen los datos reales)

1. Reemplazar `synthetic_data.py` por la ingesta del CSV real (`--csv`), y completar la
   ficha de caracterización (institución, período, variables disponibles, proporción de
   clases) tal como pide la Fase 2 de la metodología.
2. Ajustar el diccionario de datos y las reglas de inclusión/exclusión según las
   variables reales entregadas por el hospital (las columnas sintéticas actuales son
   solo un marcador de posición razonable).
3. Si el conjunto real tiene fecha índice y seguimiento, activar la interpretación como
   "riesgo futuro dentro de un horizonte" en lugar de "clasificación probabilística de
   la condición registrada" (ver `config.py` y Marco Conceptual, Sección V-C).
4. Migrar `src/data_layer/storage.py` de Parquet local a conectores reales de Snowflake
   (la interfaz `write_layer`/`read_layer` ya está pensada para ese reemplazo).
5. Añadir pruebas funcionales (matriz de casos: entrada válida, faltante, categoría
   desconocida, formato incorrecto, fuera de rango) como exige la Sección VI-C Fase 4.8.
