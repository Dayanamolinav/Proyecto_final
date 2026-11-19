# Portal institucional OncoRisk

## Diseño

Portada editorial a pantalla completa: tipografía grande (Archivo e Instrument Sans), fondo claro, un solo color de acento (azul `#2048ff`) y videos y fotografías reales en movimiento. No usa tarjetas, etiquetas en píldora ni cajas decorativas; las secciones se separan con líneas finas y fotografía. La maqueta de referencia está en `maquetas/portada.html`.

| Sección | Contenido y movimiento |
|---|---|
| Portada | Video de células a pantalla completa con el nombre del proyecto; el video se acerca y se desplaza más lento que la página (parallax) |
| Proyecto | Frase que se ilumina palabra por palabra al desplazarse |
| Cifras | Foto que se revela como cortina y cifras del modelo persistido que cuentan hacia arriba (pacientes, AUROC, Brier, sensibilidad) |
| Trazabilidad | Video a todo lo ancho con parallax y cinta de texto con las capas del proceso |
| Método | Imagen fija a la izquierda que cambia con cada uno de los cuatro pasos |
| Explora un caso | Formulario de Streamlit con estilo de hoja (solo líneas) y resultado en número grande con la posición del umbral |
| Modelo | Contribuciones globales y desempeño en la prueba reservada |
| Cierre | Video y enlace al análisis estadístico |

Los efectos respetan la preferencia de movimiento reducido del sistema y la página se adapta a pantallas pequeñas.

## Implementación

- `app/streamlit_app.py` arma las secciones con `st.html` y un único script de movimiento (`unsafe_allow_javascript=True`). DOMPurify descarta un `<script>` cuyo texto contenga etiquetas HTML, por eso el script construye los elementos con la API del DOM.
- El formulario, el resultado, la interpretación, el panel de administración y el panel estadístico siguen siendo componentes de Streamlit; solo cambia su estilo (`app/portal.css`).
- Los videos y fotos se sirven como archivos estáticos desde `app/static/media` (`server.enableStaticServing = true` en `.streamlit/config.toml`), sin incrustarlos en cada recarga.

## Recursos visuales

Videos y fotografías de [Pexels](https://www.pexels.com) (licencia de uso gratuito, sin atribución obligatoria; se acredita en el pie de página). Los videos se recortaron a 720p sin audio y se recomprimieron (unos 8 MB en total).

| Archivo | Fuente en Pexels |
|---|---|
| `v13_celulas.mp4` | Microscopic view of blue glowing cells (29851224) |
| `v7_microscopios.mp4` | Female scientists in laboratory with microscopes (32402606) |
| `v2_tubo.mp4` | Hands in gloves holding test tube (13532394) |
| `v17_medica.mp4` | A lady doctor using a laptop for her work (5998385) |
| `f23_cientifica.jpg` | Female laboratory scientist looking at a microscope (4031525) |
| `f27_microscopio.jpg` | Medical professional using microscope (9574505) |

Ninguna imagen muestra personal del proyecto ni pacientes.

## Comprobaciones

Las pruebas automatizadas de la aplicación (`tests/test_app.py`, `tests/test_statistics_app.py`, `tests/test_admin_panel.py`) verifican la predicción del formulario frente al modelo persistido, las ocho vistas del panel estadístico y el panel de administración. La portada se revisó en navegador a 1440 × 900: los cinco videos se reproducen, el script de movimiento se ejecuta y el formulario calcula la estimación dentro del nuevo diseño.
