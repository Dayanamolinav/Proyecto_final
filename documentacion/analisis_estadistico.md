# Análisis Estadístico e Inferencial

La nueva sección se abre desde el menú **Análisis estadístico**, o en la dirección local `/?vista=estadisticas`. La portada institucional y la exploración individual se conservan.

## Evidencia y granularidad

La fuente del panel es `data/synthetic/pacientes_sinteticos_20000.csv`, incluida en la entrega, no una conexión en vivo a Snowflake. Contiene 20.000 pacientes ficticios únicos, 15 columnas originales, sin celdas ausentes ni identificadores repetidos. Se derivan los mismos grupos de edad e IMC que en el prototipo. No se imputa, redondea la edad decimal, elimina atípicos ni corrige la fuente.

SHA256: `bf2078859176c2292618f8688944f8d14b2ab4688754c3bdd5ac87e4b2bc652a`.

Cifras de referencia de la base: 4.988 etiquetas 1 / 20.000 = 24,94%; Wilson95% [24,3452175894%; 25,5444072577%]. Edad media 55,076595 años y mediana 55 años. Hay 3.567 registros con sexo M y paridad > 0, producto de la simulación independiente de campos; se señalan para revisión y se conservan. Estos resultados no describen una población hospitalaria.

## Cobertura

| Vista | Cálculos y visuales |
|---|---|
| Demografía | Recuento de pacientes, etiqueta simulada e IC, edad, completitud; sexo, grupos etarios, composición por etiqueta y procedencia; barras agrupadas de edad y sexo |
| Distribuciones | Histograma configurable, distribución acumulada empírica, boxplot, Q–Q, KDE gaussiana, violines por grupo, resumen de seis variables, pruebas Shapiro–Wilk |
| Correlaciones | Dispersión con muestra visual reproducible, densidad bidimensional con toda la selección, mapa de calor Pearson/Spearman, tabla de pares y p/q |
| Contrastes y OR | Proporciones de grupo e IC de Wilson, Welch, U de Mann–Whitney, g de Hedges y efecto biserial; chi²/Fisher, V de Cramér; OR crudas y forest plot con IC95% |
| Evolución por semestre | Volumen por periodo, etiqueta con IC, cajas, medias con IC t y media móvil ponderada de dos semestres consecutivos |
| Multivariante | PCA estandarizado, varianza, proyección y pesos; K-means reproducible, tamaño y centros de grupos, silueta aproximada |
| Evaluación del modelo | Prueba reservada original verificada, matriz de confusión, ROC, Precision–Recall, calibración, Accuracy/Precision/Recall/F1, AUROC y Brier con IC bootstrap |
| Calidad y datos | Ausencias, duplicados, dominios, atípicos IQR y consistencia; inspección por ID; exportación de pacientes, filtros, procedencia y métodos |

## Definiciones y decisiones

Todos los indicadores de pacientes, sus gráficos y descargas usan la misma intersección de filtros de edad, sexo, etiqueta, fuente y semestre. Una selección vacía tiene estado explícito; Restablecer recupera la cohorte completa. Los porcentajes por grupo usan las etiquetas válidas de ese grupo, no el total de la base. Los valores p se muestran en notación científica sin redondearlos antes a cero. Benjamini–Hochberg se aplica dentro de cada familia: correlaciones, seis Welch, seis Mann–Whitney, asociaciones categóricas, seis Shapiro y todos los contrastes de OR. Los IC95% son puntuales y no simultáneos.

Las medias usan DE muestral y t; la diferencia 1−0 usa Welch–Satterthwaite. Hedges g emplea la corrección gamma del sesgo. U compara rangos/distribuciones, no necesariamente medianas. Las OR no se ajustan por otras variables; se identifica su referencia y se aplica Haldane–Anscombe (+0,5) si hay ceros. Los coeficientes del modelo penalizado no se reutilizan como estimadores inferenciales ni se inventan IC de sus coeficientes.

Para normalidad se utiliza Shapiro hasta 5.000 observaciones (submuestra reproducible cuando hay más). No se usa un KS normal con media/DE estimadas y p sin calibrar. No se selecciona automáticamente Welch o Mann–Whitney según un test previo de normalidad. Una prueba significativa con n grande no equivale a una diferencia importante.

Los diagramas de dispersión y PCA muestran hasta 2.000 puntos para legibilidad, semilla 42; sus ajustes y agregados utilizan todos los casos válidos. PCA/K-means excluyen únicamente casos incompletos para esos cálculos, estandarizan seis variables y no usan la etiqueta para formar grupos. La silueta usa hasta 1.000 casos. La KDE se calcula sobre todos los valores válidos de cada grupo, con banda de Scott; el ancho de un violín representa densidad, no tamaño del grupo. La paridad es discreta: conviene priorizar sus frecuencias.

Los cuatro semestres no son consultas fechadas: cada paciente tiene una observación. La media móvil combina las filas de dos semestres consecutivos y se pondera por n; no promedia medias sin ponderar. Los periodos ausentes no se interpolan. No son estimables el seguimiento individual, incidencia, supervivencia Kaplan–Meier, Cox, efectos de tratamientos ni pronósticos fiables con esta estructura.

## Población del modelo

Desde la versión del 17 de noviembre de 2026, el modelo se entrena con este mismo lote de 20.000 pacientes. Su evaluación utiliza la partición reservada de 4.000 pacientes de `data/analytical/clinical_analytical.parquet`. Se reconstruye con la misma semilla y estratificación agrupada por paciente, cotejando AUROC y las cuatro celdas de la matriz (con el umbral guardado junto al modelo, 0,25) contra la evaluación persistida. No se vuelve a ajustar ni seleccionar ningún modelo. Si no coincide, se muestra un error en lugar de adjudicarle una prueba falsa. El control deslizante del umbral parte del valor elegido en desarrollo y solo sirve para explorar.

Los filtros del CSV no modifican esa prueba independiente del panel de cohortes; la interfaz indica esta excepción. El umbral es exploratorio: no se optimiza con la prueba ni se propone para decisiones clínicas. Los IC95% del modelo usan bootstrap percentil estratificado, 500 réplicas y semilla 42. Son condicionales al modelo fijo y la proporción de clases observada; no incluyen incertidumbre de entrenamiento ni son validación externa. Las métricas sin denominador se muestran como no definidas.

## Etiquetas y reproducción

`VARIABLE_LABELS` y las traducciones de resultados se aplican a títulos, ejes, leyendas, tooltips, tablas y variables codificadas del modelo. Las descargas de pacientes conservan el esquema técnico original para permitir reutilización; las tablas de la página usan nombres legibles.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app/streamlit_app.py
```

Las funciones reproducibles están en `src/analytics/`; la interfaz está en `app/statistics_panel.py`. No se cambiaron las capas de entrenamiento ni almacenamiento del prototipo. Se añadieron pruebas de resultados conocidos, fórmulas, estados límite y filtros.

## Documentación técnica

- [Welch y contraste t en SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
- [U de Mann–Whitney en SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
- [Funciones estadísticas y control FDR en SciPy](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [PCA en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Estandarización en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [K-means en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
