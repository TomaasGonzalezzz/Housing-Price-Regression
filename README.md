# Housing Price Regression
### Regresión lineal, Ridge y Lasso con NumPy

Predicción de precios de viviendas a partir de sus características. El trabajo compara modelos de complejidad creciente y estudia qué aportan la ingeniería de variables, la regularización y la selección mediante validación.

**Tomás González** · Machine Learning, UdeSA · 2026.

![Predicciones del modelo final sobre test](assets/predicciones-test.png)

[Notebook](notebook/Entrega_TP1.ipynb) · [Consigna](docs/consigna.pdf)

## Modelado y evaluación

Implementé los modelos con NumPy: solución por pseudoinversa, descenso por gradiente y penalizaciones L1/L2. El desarrollo incluye limpieza, conversión de unidades, construcción de variables, comparación de seis modelos y curvas de aprendizaje.

El modelo seleccionado en la entrega es M5. Tras reentrenarlo sobre desarrollo, los resultados guardados de test son:

| MAE | RMSE | R² |
| ---: | ---: | ---: |
| 19.723,00 USD | 29.975,85 USD | 0,8933 |

Los datos originales contienen 1.600 registros de desarrollo y 400 de test. Tras la limpieza quedan 1.488 y 372, respectivamente. Estas métricas proceden de la ejecución original, no de un entrenamiento nuevo.

## Ejecutar

Crear un entorno virtual e instalar requirements.txt. El notebook registra Python 3.12.2; las versiones originales de bibliotecas no quedaron fijadas.

~~~sh
python -m pip install -r requirements.txt
cd notebook
python -m jupyterlab
~~~

Abrir Entrega_TP1.ipynb y mantener el directorio del kernel en **notebook/**. Los módulos están en **src/**, y los datos originales y procesados en **data/**. La ejecución completa puede tardar y reescribe CSV intermedios.

**Herramientas:** Python, NumPy, pandas, Matplotlib y Seaborn. Los modelos no utilizan scikit-learn.

[Documentación técnica y alcance de las verificaciones](DOCUMENTACION.md).
