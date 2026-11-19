# Matriz de pruebas funcionales

Fecha: 13 de noviembre de 2026. Versión: 1.0. Responsable de la ejecución: Codex. Entorno: Python 3.14, datos sintéticos y modelo generado por el pipeline corregido. Estas pruebas verifican software y no constituyen validación clínica.

Precondición general: instalar las dependencias; para el caso PF-08 ejecutar primero el pipeline sintético. Ejecución: `python -m unittest discover -s tests -v` desde la raíz. Evidencia: 13 pruebas aprobadas en la copia integrada, incluida la separación por paciente.

| ID | Entrada | Resultado esperado | Resultado obtenido | Estado |
|---|---|---|---|---|
| PF-01 | Datos completos y categorías válidas | Prepara los datos y variables derivadas | Se prepararon correctamente | Aprobado |
| PF-02 | Edad ausente | Rechaza la entrada y señala el campo | ValueError sobre edad | Aprobado |
| PF-03 | Tabaquismo desconocido | Rechaza la categoría y señala el campo | ValueError sobre tabaquismo | Aprobado |
| PF-04 | Edad con texto `abc` | Rechaza el formato | ValueError sobre edad | Aprobado |
| PF-05 | Edad 150 | Rechaza el valor fuera del límite de la interfaz | ValueError sobre edad | Aprobado |
| PF-06 | Edad 18 y 100 | Acepta los límites de la interfaz | Ambos valores aceptados | Aprobado |
| PF-07 | Edad NaN, infinita o fraccionaria | Rechaza valores no válidos | Tres valores rechazados | Aprobado |
| PF-08 | Formulario con valores predeterminados | La salida coincide con `predict_proba` del modelo persistido | Coincide, tras redondear a un decimal porcentual | Aprobado |
| PF-09 | Edades 39, 40, 54, 55, 69 y 70 | Los grupos de edad corresponden a sus etiquetas | Las seis edades se agruparon correctamente | Aprobado |
| PF-10 | IMC 18,4; 18,5; 24,9; 25; 29,9; 30 | Los grupos de IMC corresponden a sus límites | Las seis entradas se agruparon correctamente | Aprobado |

PF-02 a PF-07 prueban la función de validación que utiliza el formulario, no la interacción visual de controles que ya restringen sus opciones. El modelo de aprendizaje contiene un imputador y un codificador que puede ignorar categorías desconocidas; la política del formulario exige entradas completas y categorías conocidas antes de invocarlo. Los límites son técnicos del prototipo sintético y deberán revisarse cuando se conozcan los datos reales. Si falta el modelo, PF-08 se omite y debe volver a ejecutarse tras el entrenamiento.
