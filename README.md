# Prototipo académico de análisis oncológico

Este proyecto demuestra una canalización de ingeniería de datos y aprendizaje automático con **datos sintéticos**. No contiene datos del hospital, no estima riesgo clínico validado y no debe usarse para diagnóstico ni decisiones clínicas.

## Preparación y ejecución

Requiere Python, pip y las dependencias de `requirements.txt`. Desde la carpeta del proyecto:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python -m streamlit run app/streamlit_app.py
```

El pipeline genera datos de ejemplo, construye capas locales RAW, CLEAN y ANALYTICAL, selecciona entre regresión logística, Random Forest y XGBoost mediante validación cruzada en desarrollo y evalúa el modelo seleccionado sobre la prueba reservada. Produce un modelo en `models/`, informes en `reports/` y bitácora en `logs/`. El cálculo depende de las versiones instaladas; vuelva a entrenar si no puede cargar el modelo distribuido.

Para ejecutar las pruebas:

```bash
python -m unittest discover -s tests -v
```

La prueba de la aplicación requiere que exista el modelo entrenado. Consulte `MATRIZ_PRUEBAS.md` para los casos y `documentacion/` para la ficha, el diccionario, las limitaciones y el plan de datos reales.

## Alcance y decisiones

- La separación y la validación cruzada se agrupan por identificador de paciente. El desenlace debe ser consistente para cada paciente y ambas clases deben estar presentes. El conjunto de prueba se usa solo para el modelo seleccionado.
- Las columnas sintéticas `nombre` y `documento_identidad` se eliminan antes de RAW. Esto no garantiza eliminar todos los identificadores de un archivo real.
- El formulario requiere valores completos, tipos válidos y categorías conocidas. Las probabilidades y variables importantes describen únicamente el modelo sintético; no demuestran causalidad.
- Los datos se almacenan localmente en Parquet. No se ha implementado Snowflake.
- La ingesta de CSV está deshabilitada hasta definir el esquema, la variable objetivo, la temporalidad, los controles de identificación y las autorizaciones de la fuente real. No basta con pasar `--csv`.
- Los pesos de clase y el umbral son decisiones técnicas; las probabilidades pueden estar mal calibradas. La validación es interna y experimental.

La ficha de caracterización conserva cifras de referencia identificadas como pertenecientes a la ejecución anterior. Los resultados actuales están en `reports/resumen_pipeline.json`; no deben interpretarse como resultados de pacientes reales.