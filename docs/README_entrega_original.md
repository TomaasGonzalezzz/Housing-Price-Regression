# Trabajo Practico 1 - I302

Esta carpeta contiene la entrega del TP1 de regresion. La idea del trabajo es ir desde un analisis inicial del dataset hasta la comparacion final entre modelos lineales, con una implementacion propia de OLS, Ridge y Lasso.

## Como esta organizado

- `notebook/Entrega_TP1.ipynb`
  Notebook principal. Sigue el orden de la consigna y muestra el desarrollo completo, las tablas, los graficos y las conclusiones.
- `src/models.py`
  Implementacion de la clase `RegresionLineal`. Aca esta la parte central del TP: pseudo-inversa, descenso por gradiente, Ridge y Lasso.
- `src/metrics.py`
  Funciones de metricas. Se usan en todo el notebook para comparar modelos con ECM, MAE, RMSE y R2.
- `src/data_splitting.py`
  Helpers para separar train/validation y para armar folds de validacion cruzada.
- `src/preprocessing.py`
  Limpieza de datos, conversion de unidades, imputacion y feature engineering. Tambien define las columnas finales de M4 y M5.
- `src/utils.py`
  Funciones puente entre notebook y modulos base. Aca quedaron helpers de entrenamiento, evaluacion, regularizacion, curvas de aprendizaje y exportes auxiliares.
- `src/visualization.py`
  Todas las funciones de graficos del TP. Separamos esto para no cargar el notebook con mucho codigo visual.
- `data/raw/casas_dev.csv`
  Dataset de desarrollo provisto por la catedra.
- `data/raw/casas_test.csv`
  Dataset de test provisto por la catedra.
- `data/processed/`
  Carpeta donde el notebook deja algunos CSV intermedios cuando se ejecuta.

## Recorrido del trabajo

1. Primero se cargan `casas_dev.csv` y `casas_test.csv` y se hace una inspeccion general del dataset.
2. Despues se construye un dataset base: se crea `zona`, se convierten superficies a m2, se limpian precios invalidos y se agregan algunas columnas simples.
3. Con el dataset base listo, `casas_dev.csv` se divide en train y validation. Las imputaciones de `pisos` y `edad` se aprenden solo con train.
4. A partir de ahi se implementa la clase `RegresionLineal` y se verifica que la solucion por pseudo-inversa y por gradiente den resultados consistentes.
5. Luego se entrenan los modelos pedidos por la consigna: M1, M2 y M3.
6. En el punto de feature engineering se construyen nuevas variables y se definen M4 y M5.
7. En regularizacion se toma M4 como base y se comparan Ridge y Lasso para definir M6.
8. Al final se comparan learning curves, se elige el modelo final y se reporta el resultado sobre test.

## Que hace cada modulo

### `src/models.py`

La clase `RegresionLineal` recibe `X` e `y` y guarda los coeficientes en `self.coef`.

Incluye:

- ajuste por pseudo-inversa para OLS;
- ajuste analitico para Ridge;
- ajuste iterativo para Lasso;
- prediccion;
- tabla e impresion de coeficientes;
- conteo de coeficientes casi nulos.

### `src/metrics.py`

Agrupa las metricas de regresion usadas en todo el TP:

- `mse()`
- `mae()`
- `rmse()`
- `r2()`
- `resumir_metricas_regresion()`

### `src/data_splitting.py`

Tiene dos funciones cortas:

- `train_val_split()`: arma el split 80/20 con semilla fija;
- `generar_folds_kfold()`: devuelve los indices para validacion cruzada.

### `src/preprocessing.py`

Este archivo concentra casi todo el trabajo previo al modelado:

- construccion del dataset base;
- imputacion de `pisos` y `edad`;
- creacion de variables derivadas;
- definicion de columnas para M4 y M5;
- armado de matrices listas para train, validation y test.

### `src/utils.py`

Quedaron funciones practicas para no repetir codigo en el notebook:

- preparar matrices para modelar;
- entrenar y resumir modelos;
- construir rankings de features;
- evaluar regularizacion para distintas grillas de `lambda`;
- armar learning curves;
- exportar datasets auxiliares para inspeccion.

### `src/visualization.py`

Reune todos los graficos usados en la entrega:

- EDA;
- comparacion de metricas;
- regularizacion;
- importancia de coeficientes;
- learning curves;
- residuos y paneles de prediccion.

## Nota sobre el uso de IA

En esta entrega la IA se uso como apoyo de presentacion, sobre todo para pensar variantes de graficos y para ordenar mejor algunos displays y tablas del notebook. Eso se refleja principalmente en `src/visualization.py` y en la forma en que se muestran varios resultados en `Entrega_TP1.ipynb`.

La idea fue usarla para emprolijar la comunicacion visual de la entrega y hacer mas clara la lectura de resultados. La implementacion del flujo de datos, las metricas y los modelos queda en el codigo del proyecto y no usa librerias de machine learning externas.

## Librerias usadas

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `pathlib`
- `IPython.display`

## Ejecucion

1. Abrir `notebook/Entrega_TP1.ipynb`.
2. Verificar que la carpeta `data/raw/` tenga `casas_dev.csv` y `casas_test.csv`.
3. Ejecutar las celdas en orden.

