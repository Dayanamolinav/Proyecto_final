# OncoRisk · Prototipo académico de análisis oncológico

Este proyecto demuestra una canalización de ingeniería de datos y aprendizaje automático con **datos sintéticos** (20.000 pacientes ficticios). No contiene datos del hospital, no estima riesgo clínico validado y no debe usarse para diagnóstico ni decisiones clínicas.

## Qué incluye

| Componente | Dónde |
|---|---|
| Snowflake: capas RAW/CLEAN/ANALYTICAL, Stage, auditoría | `sql/00_setup_inicial.sql` |
| Ingesta del CSV por Stage (PUT + COPY INTO) con registro de cargas | `sql/01_ingesta_stage.sql`, `scripts/cargar_csv_snowflake.py` |
| Transformaciones SQL RAW → CLEAN → ANALYTICAL con bitácora | `sql/02_raw_a_clean.sql`, `sql/03_clean_a_analytical.sql` |
| Las mismas transformaciones en Python, para trabajar sin conexión | `src/processing/` |
| Cuaderno de análisis exploratorio | `notebooks/01_analisis_exploratorio.ipynb` |
| Modelado: regresión logística, Random Forest y XGBoost; validación cruzada agrupada por paciente; umbral elegido en desarrollo | `src/modeling/` |
| Portal web en Streamlit: portada con videos, formulario, interpretación y panel estadístico de 8 vistas | `app/streamlit_app.py`, `app/portal.css`, `app/static/media` (`documentacion/redisenio_portal.md`) |
| Contenedor Docker | `Dockerfile`, `docker-compose.yml` |
| Pruebas automáticas (47) y matriz de pruebas | `tests/`, `MATRIZ_PRUEBAS.md` |
| Ficha, diccionario, decisiones, resultados y plan de datos reales | `documentacion/` |

## Ejecutar en local (sin Snowflake)

Requiere Python 3.11 o superior. Desde la carpeta del proyecto:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --backend local --usar-csv-local
python -m streamlit run app/streamlit_app.py
```

En Windows también puede abrir la página con `ABRIR_PAGINA.bat`. El archivo de entrada es `app/streamlit_app.py`.

El pipeline lee los 20.000 pacientes de `data/synthetic/`, construye las capas RAW, CLEAN y ANALYTICAL, compara los tres modelos con validación cruzada en desarrollo, elige el umbral con predicciones fuera de pliegue y evalúa una vez el modelo seleccionado sobre la prueba reservada (20 %). Produce el modelo en `models/`, informes en `reports/` y la bitácora en `logs/`. El modelo guardado depende de las versiones instaladas: si no carga, vuelva a ejecutar el pipeline.

Otras fuentes de datos:

```bash
python scripts/run_pipeline.py --backend local                        # lee los 20.000 desde Snowflake, capas en local
python scripts/run_pipeline.py --backend snowflake                    # capas en Snowflake
python scripts/run_pipeline.py --backend snowflake --transformar-con-sql   # CLEAN y ANALYTICAL con sql/02 y sql/03
python scripts/run_pipeline.py --backend local --usar-datos-locales   # lote pequeño generado al momento
```

## Ejecutar con Docker

```bash
docker compose up --build
```

Abra http://localhost:8501. Si ese puerto está ocupado (por ejemplo, porque la página ya corre fuera de Docker), use otro: en PowerShell `$env:ONCORISK_PORT=8502; docker compose up` y abra http://localhost:8502. La imagen vuelve a entrenar el modelo y corre las pruebas al construirse, para que el modelo coincida con las versiones instaladas. El `.env` no entra en la imagen; `docker-compose.yml` lo pasa como variables si existe.

## Cuaderno exploratorio

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyterlab notebooks/01_analisis_exploratorio.ipynb
```

El cuaderno se guarda con las salidas ya ejecutadas. Si al abrirlo en otro computador las gráficas aparecen en blanco, es porque Jupyter no confía en salidas guardadas por otra persona: use **Run > Run All Cells** (necesita conexión a internet para cargar Vega). Para ejecutarlo sin abrir JupyterLab: `python -m nbconvert --to notebook --execute --inplace notebooks/01_analisis_exploratorio.ipynb`.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

Las pruebas de la aplicación y de la evaluación requieren el modelo entrenado. `tests/test_sql_transformations.py` ejecuta los scripts SQL en SQLite y comprueba que producen las mismas tablas que Python.

## Snowflake

### Credenciales

Con `SNOWFLAKE_AUTHENTICATOR=snowflake` los scripts piden la contraseña de Snowflake del usuario del `.env`. Con `PROGRAMMATIC_ACCESS_TOKEN` piden un token PAT. Si entra a Snowsight con Google u otro proveedor, su contraseña web no sirve para el conector: genere un token en **Governance & security > Users & roles > su usuario > Programmatic access tokens > Generate new token**, con un nombre como `ONCORISK_DEV` y una expiración corta. El token aparece una sola vez; guárdelo en un gestor de contraseñas.

Los scripts piden la contraseña o el token de forma oculta y lo mantienen solo durante el proceso. En el panel de administración de la página el campo cambia de nombre según el método (**Contraseña de Snowflake** o **Token PAT de Snowflake**). No escriba ninguno de los dos en `.env`, Git, el código ni el chat. Snowflake puede pedir activar temporalmente **Bypass requirement for network policy** al generarlo.

Copie `.env.snowflake.example` como `.env` y complete su usuario.

### Primera carga

```bash
python scripts/test_snowflake_connection.py                 # comprueba la conexión
python scripts/cargar_csv_snowflake.py --setup              # crea objetos (sql/00) y carga el CSV por Stage (sql/01)
python scripts/run_pipeline.py --backend snowflake --transformar-con-sql
```

`scripts/ejecutar_sql.py` ejecuta cualquier archivo de `sql/`; `python scripts/ejecutar_sql.py sql/04_verificaciones.sql` muestra el registro de cargas, las filas por capa y la bitácora (solo lectura). La carga usa una tabla temporal: si falla, la tabla de aterrizaje no se modifica. Los scripts SQL también pueden abrirse en una hoja de Snowsight, salvo el `PUT`, que necesita SnowSQL o el script de carga.

La autenticación `externalbrowser` fue probada y la cuenta respondió con el código 390190 porque no tiene un proveedor SAML configurado.

## Alcance y decisiones

- La separación y la validación cruzada se agrupan por identificador de paciente. El conjunto de prueba se usa solo para el modelo seleccionado.
- Los modelos se entrenan **sin pesos de clase** y el **umbral (0,25) se elige maximizando F1 en desarrollo**. Con esto el Brier del modelo (0,160) queda por debajo del modelo nulo (0,187). Evidencia y limitaciones en `documentacion/decisiones_modelado.md`.
- Se eliminan los valores clínicamente imposibles; los atípicos plausibles se conservan y se reportan.
- Las columnas sintéticas `nombre` y `documento_identidad` se eliminan antes de RAW. Esto no garantiza eliminar todos los identificadores de un archivo real.
- El formulario requiere valores completos, tipos válidos y categorías conocidas. Las probabilidades y variables importantes describen únicamente el modelo sintético; no demuestran causalidad.
- La ingesta de un CSV real (`--csv`) está deshabilitada hasta definir el esquema, la variable objetivo, la temporalidad, los controles de identificación y las autorizaciones de la fuente (`documentacion/plan_datos_reales.md`).
- La validación es interna y experimental. Resultados actuales: `reports/resumen_pipeline.json` y `documentacion/resultados_y_limitaciones.md`.
