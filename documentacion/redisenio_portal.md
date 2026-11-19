# Portal institucional OncoRisk

Versión del 16 de noviembre de 2026, basada en `oncorisk_conexion_snowflake.zip`. La copia anterior se conserva intacta.

## Diseño

Se aplicaron las pautas de diseño de la habilidad Sites al proyecto local existente. La referencia es el video aportado por la usuaria: portal institucional de salud, navegación blanca superior, azul y turquesa, banner fotográfico, iconos circulares y pie institucional. La marca y los textos pertenecen a OncoRisk; no se copiaron nombres, teléfonos, servicios asistenciales ni logotipos de la clínica del video.

La portada enlaza a las secciones de proyecto, herramientas, formulario, resultados, interpretación y metodología. Incluye estados iniciales, foco visible de teclado, adaptación a pantallas pequeñas y respeto por la preferencia de movimiento reducido. Los recursos visuales y las fuentes se cargan localmente.

## Funcionalidad

El formulario conserva las once entradas, sus límites, las categorías y los valores iniciales. Sigue usando `prepare_patient_input`, las dos variables derivadas y el mismo modelo persistido. Conserva el último resultado enviado y calcula una nueva estimación al enviar de nuevo el formulario. La probabilidad, la explicación global y las métricas proceden del modelo existente. Se añadió una gráfica de barras de sus contribuciones globales y un desplegable con los valores del cálculo.

Los datos continúan en Parquet local. No se migró el almacenamiento ni se verificó una cuenta de Snowflake como parte de este rediseño. El CSV de 20.000 pacientes no se carga automáticamente. No se añadieron historiales ni curvas de evolución individual.

## Ejecución

Desde la carpeta `oncorisk_prototype`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Esta copia incluye el modelo de demostración anterior. Si su entorno no puede cargarlo, regenérelo con `python scripts/run_pipeline.py`.

## Comprobaciones

Las 18 pruebas automatizadas pasaron. Incluyen la comparación de la predicción del formulario con el modelo persistido, cambios de entradas y conservación del último resultado enviado. Los 19 módulos de `src/` y el modelo se compararon por SHA256 con la versión anterior y son idénticos.

Se revisaron en navegador la portada, los enlaces al formulario y a los resultados, el cálculo y la gráfica de contribuciones. En escritorio (1365 px) y móvil (390 px) no hubo desbordamiento horizontal; el formulario móvil se distribuye en una columna. Se actualizó el requisito a la versión de Streamlit comprobada, 1.67.

## Recurso visual

Archivo: `app/assets/equipo-investigacion.png`. Se generó con la herramienta integrada de generación de imágenes, sin claves API ni fotografías de personal real. La página la identifica como ilustrativa.

Prompt utilizado:

> Use case: photorealistic-natural. Asset type: wide hero photograph for OncoRisk, a Spanish-language academic oncology analysis portal inspired by a clean blue/turquoise hospital website. Primary request: a welcoming professional healthcare research team, four fictional Latin American doctors and researchers, two women and two men aged 30-55, white lab coats with navy and teal scrubs, natural friendly expressions, standing together in a bright modern hospital research corridor. Medium: high quality realistic editorial photography, natural skin texture, soft daylight, elegant calm cyan and white palette. Composition: landscape 3:2, waist up group arranged compactly in center and right, enough headroom, softly blurred light clinical environment. No patient, no medical procedures, no identifiable real person, no hospital logos, no text, no letters, no watermark. Intended as illustrative decorative photo, not claims about actual staff.
