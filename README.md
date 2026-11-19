# Prototipo académico de análisis oncológico

Este proyecto demuestra una canalización de ingeniería de datos y aprendizaje automático con **datos sintéticos**. No contiene datos del hospital, no estima riesgo clínico validado y no debe usarse para diagnóstico ni decisiones clínicas.

## Preparación y ejecución

Esta copia incluye el portal institucional inspirado en el video de referencia del 16 de noviembre de 2026: navegación superior, portada azul y turquesa, accesos por sección, formulario y resultados. Conserva los módulos de procesamiento, entrenamiento, validación y conexión a Snowflake de la versión anterior. El rediseño no cambia la fuente de datos ni conecta automáticamente Snowflake.

El archivo de entrada de la página es `app/streamlit_app.py` (no `app.py`). Consulta `documentacion/redisenio_portal.md` para ver los cambios y las comprobaciones.

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
## Conexión con Snowflake

El proyecto incluye una prueba de conexión de solo lectura. La configuración usa variables de entorno y no guarda la contraseña en código ni archivos.

En PowerShell, desde la raíz del proyecto:

```powershell
$env:SNOWFLAKE_ACCOUNT = "TWGQDNO-YI73415"
$env:SNOWFLAKE_USER = "SU_USUARIO_DE_SNOWFLAKE"
$env:SNOWFLAKE_ROLE = "ACCOUNTADMIN"
$env:SNOWFLAKE_AUTHENTICATOR = "PROGRAMMATIC_ACCESS_TOKEN"
python scripts/test_snowflake_connection.py
```

El script solicita la contraseña de forma oculta y la mantiene solo durante el proceso. Al conectarse, muestra el contexto activo y los warehouses disponibles. Después se podrán definir `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE` y `SNOWFLAKE_SCHEMA`.

La autenticación `externalbrowser` fue probada y la cuenta respondió con el código 390190 porque no tiene un proveedor SAML configurado. Si la cuenta habilita SSO en el futuro, puede cambiarse `SNOWFLAKE_AUTHENTICATOR` a `externalbrowser`.

### Autenticación recomendada para la aplicación: token PAT

Si el acceso web utiliza Google u otro proveedor, la contraseña web puede no ser una contraseña nativa válida para el conector. Genere un token en Snowsight desde **Governance & security > Users & roles > su usuario > Programmatic access tokens > Generate new token**. Use un nombre como `ONCORISK_DEV`, restríngalo a un rol dedicado cuando ese rol exista y defina una expiración corta durante el desarrollo.

El token aparece una sola vez. Cópielo a un gestor de secretos y péguelo en el prompt del script como si fuera la contraseña. No lo escriba en `.env`, Git, el código ni el chat. Para usuarios humanos sin política de red, Snowflake puede pedir activar temporalmente **Bypass requirement for network policy** al generar el token.
