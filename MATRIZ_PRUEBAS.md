# Matriz de pruebas funcionales

Fecha: 17 de noviembre de 2026. Versión: 1.1. Entorno: Python 3.14, scikit-learn 1.9.1, Streamlit 1.67, datos sintéticos de 20.000 pacientes y modelo generado con `python scripts/run_pipeline.py --backend local --usar-csv-local`. Estas pruebas verifican software y no constituyen validación clínica.

Ejecución: `python -m unittest discover -s tests -v` desde la raíz. Evidencia: **47 pruebas automatizadas aprobadas** más cuatro pruebas manuales (PF-20 a PF-22 en Snowflake y PF-23 en JupyterLab), incluidas la separación por paciente, la equivalencia SQL–Python y el panel de administración. Las pruebas PF-08, PF-13 y PF-14 se omiten si falta el modelo; deben repetirse después de entrenar.

| ID | Precondición | Entrada | Resultado esperado | Resultado obtenido | Estado |
|---|---|---|---|---|---|
| PF-01 | Ninguna | Datos completos y categorías válidas | Prepara los datos y variables derivadas | Se prepararon correctamente | Aprobado |
| PF-02 | Ninguna | Edad ausente | Rechaza la entrada y señala el campo | ValueError sobre edad | Aprobado |
| PF-03 | Ninguna | Tabaquismo desconocido | Rechaza la categoría y señala el campo | ValueError sobre tabaquismo | Aprobado |
| PF-04 | Ninguna | Edad con texto `abc` | Rechaza el formato | ValueError sobre edad | Aprobado |
| PF-05 | Ninguna | Edad 150 | Rechaza el valor fuera del límite de la interfaz | ValueError sobre edad | Aprobado |
| PF-06 | Ninguna | Edad 18 y 100 | Acepta los límites de la interfaz | Ambos valores aceptados | Aprobado |
| PF-07 | Ninguna | Edad NaN, infinita o fraccionaria | Rechaza valores no válidos | Tres valores rechazados | Aprobado |
| PF-08 | Modelo entrenado | Formulario con valores predeterminados y luego editados | La salida coincide con `predict_proba` del modelo persistido | Coincide, tras redondear a un decimal porcentual | Aprobado |
| PF-09 | Ninguna | Edades 39, 40, 54, 55, 69 y 70 | Los grupos de edad corresponden a sus etiquetas | Las seis edades se agruparon correctamente | Aprobado |
| PF-10 | Ninguna | IMC 18,4; 18,5; 24,9; 25; 29,9; 30 | Los grupos de IMC corresponden a sus límites | Las seis entradas se agruparon correctamente | Aprobado |
| PF-11 | Ninguna | Lote con duplicados, faltantes, valores imposibles, columnas con fuga y 3 registros sin objetivo | `sql/02` y `sql/03` producen las mismas filas que `cleaning.py` y `feature_engineering.py`; la bitácora SQL cuenta cada exclusión | Tablas idénticas; conteos de duplicados, imposibles y sin objetivo correctos | Aprobado |
| PF-12 | CSV sintético local | Lote de 20.000 pacientes | SQL y Python producen la misma tabla ANALYTICAL | Tablas idénticas | Aprobado |
| PF-13 | Modelo entrenado | Modelo persistido | El umbral se eligió con el criterio de desarrollo y es el mismo que se usó en la prueba | `max_f1_fuera_de_pliegue_desarrollo`; umbral 0,25 en ambos | Aprobado |
| PF-14 | Modelo entrenado | Prueba reservada reconstruida | Reproduce la matriz de confusión y el AUROC guardados con el umbral persistido | Coincide (4.000 pacientes) | Aprobado |
| PF-15 | Ninguna | Probabilidades y etiquetas de ejemplo | El umbral maximiza F1; en empate elige el más bajo; rechaza predicciones faltantes | Comportamiento esperado | Aprobado |
| PF-16 | Ninguna | Validación cruzada sobre 200 pacientes | Cada paciente de desarrollo recibe una predicción fuera de pliegue; 5 AUROC por modelo | Sin faltantes; 5 pliegues | Aprobado |
| PF-17 | `ADMIN_PANEL_CODE` vacío; autenticación PAT | Panel admin, fuente Snowflake, token escrito | El token se entrega como `SNOWFLAKE_TOKEN` (no como contraseña) | `SNOWFLAKE_TOKEN` asignado; `SNOWFLAKE_PASSWORD` vacío | Aprobado |
| PF-18 | `ADMIN_PANEL_CODE` vacío | Panel admin, lote local pequeño | Reentrena sin conectarse a Snowflake | `backend="local"`, `use_snowflake_csv=False` | Aprobado |
| PF-19 | `ADMIN_PANEL_CODE` vacío; autenticación PAT | Panel admin, fuente Snowflake, sin token | Muestra un error y no entrena | "Ingresa el token PAT de Snowflake."; no se llamó al pipeline | Aprobado |
| PF-20 | Snowflake con `RAW.CLINICAL_RAW` (20.000) y copia `CLONE` de la ANALYTICAL escrita por Python | `sql/00`, `sql/02` y `sql/03` en Snowsight; `MINUS` en ambos sentidos | Objetos creados; ANALYTICAL idéntica a la de Python; bitácora con 5 pasos | 0 filas distintas en cada sentido; 20.000 filas; 0 nulos obligatorios; 0 repetidos; 5 registros en `AUDITORIA.BITACORA_TRANSFORMACIONES` | Aprobado |
| PF-21 | Objetos de `sql/00`; CSV sintético local | `python scripts/cargar_csv_snowflake.py` (PUT + `sql/01`) y `sql/04_verificaciones.sql` | El CSV entra por el Stage, se registra la carga y la tabla de aterrizaje queda igual a la capa RAW | Registro LOADED con 20.000 leídas y 20.000 cargadas, sin error; 20.000 filas en las cuatro capas; 0 diferencias con `RAW.CLINICAL_RAW` en ambos sentidos | Aprobado |
| PF-22 | Objetos y lote cargados en Snowflake; contraseña del usuario | `python scripts/run_pipeline.py --backend snowflake --transformar-con-sql` | Lee el lote en Snowflake, construye CLEAN y ANALYTICAL con SQL allá y entrena; mismos resultados que la corrida local | Origen `RAW.PACIENTES_PRUEBA_SINTETICOS`; bitácora con `transformaciones_sql_snowflake`; misma regresión logística, umbral 0,25 y métricas de prueba idénticas; 47 pruebas aprobadas con ese modelo | Aprobado |
| PF-23 | `requirements-notebook.txt` instalado | `notebooks/01_analisis_exploratorio.ipynb` ejecutado con nbconvert y abierto en JupyterLab | Todas las celdas se ejecutan sin error y las gráficas se muestran | 14 de 14 celdas ejecutadas, 0 errores; gráficas visibles en JupyterLab 4.6 | Aprobado |

PF-02 a PF-07 prueban la función de validación que utiliza el formulario, no la interacción visual de controles que ya restringen sus opciones. El modelo de aprendizaje contiene un imputador y un codificador que puede ignorar categorías desconocidas; la política del formulario exige entradas completas y categorías conocidas antes de invocarlo. PF-11 y PF-12 ejecutan los scripts SQL en SQLite. Además, PF-20 y PF-21 se ejecutaron en la cuenta de Snowflake del proyecto. Los límites son técnicos del prototipo sintético y deberán revisarse cuando se conozcan los datos reales.
