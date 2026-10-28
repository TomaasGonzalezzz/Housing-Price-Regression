"""Helpers de entrenamiento y comparacion."""

from pathlib import Path

import numpy as np
import pandas as pd

from data_splitting import generar_folds_kfold
from metrics import mae, mse, rmse, r2, resumir_metricas_regresion
from models import RegresionLineal


def preparar_features_para_modelo(df_train_modelo, df_validation_modelo, columnas_numericas, columnas_categoricas):
    """Arma X_train y X_validation usando solo info de train."""
    X_train_numerico = df_train_modelo[columnas_numericas].copy()
    X_validation_numerico = df_validation_modelo[columnas_numericas].copy()
    medias_train = X_train_numerico.mean(axis=0)
    desvios_train = X_train_numerico.std(axis=0, ddof=0).replace(0, 1)
    X_train_numerico_escalado = (X_train_numerico - medias_train) / desvios_train
    X_validation_numerico_escalado = (X_validation_numerico - medias_train) / desvios_train

    if columnas_categoricas:
        columnas_dummy = pd.get_dummies(df_train_modelo[columnas_categoricas], drop_first=True, dtype=int).columns.tolist()

        def alinear_dummies(dataframe):
            dummies = pd.get_dummies(dataframe[columnas_categoricas], drop_first=True, dtype=int)
            for columna in columnas_dummy:
                if columna not in dummies.columns:
                    dummies[columna] = 0
            columnas_extra = [columna for columna in dummies.columns if columna not in columnas_dummy]
            if columnas_extra:
                dummies = dummies.drop(columns=columnas_extra)
            return dummies[columnas_dummy]

        X_train_categorico = alinear_dummies(df_train_modelo)
        X_validation_categorico = alinear_dummies(df_validation_modelo)
    else:
        X_train_categorico = pd.DataFrame(index=df_train_modelo.index)
        X_validation_categorico = pd.DataFrame(index=df_validation_modelo.index)

    X_train = pd.concat([X_train_numerico_escalado.reset_index(drop=True), X_train_categorico.reset_index(drop=True)], axis=1)
    X_validation = pd.concat(
        [X_validation_numerico_escalado.reset_index(drop=True), X_validation_categorico.reset_index(drop=True)],
        axis=1,
    )

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_train_numerico_escalado": X_train_numerico_escalado,
        "X_validation_numerico_escalado": X_validation_numerico_escalado,
        "X_train_categorico": X_train_categorico,
        "X_validation_categorico": X_validation_categorico,
    }


def _resumir_metricas_train_y_validation(y_train, pred_train, y_validation, pred_validation):
    """Devuelve el mismo bloque de metricas para train y validation."""
    resumen_train = resumir_metricas_regresion(y_train, pred_train)
    resumen_validation = resumir_metricas_regresion(y_validation, pred_validation)
    return {
        "ECM_train": resumen_train["ECM"],
        "MAE_train": resumen_train["MAE"],
        "RMSE_train": resumen_train["RMSE"],
        "R2_train": resumen_train["R2"],
        "ECM_validation": resumen_validation["ECM"],
        "MAE_validation": resumen_validation["MAE"],
        "RMSE_validation": resumen_validation["RMSE"],
        "R2_validation": resumen_validation["R2"],
    }


def entrenar_y_resumir_modelo(
    nombre_modelo,
    escenario_target,
    X_train,
    y_train,
    X_validation,
    y_validation,
    zonas_validation=None,
):
    modelo = RegresionLineal(X_train.values, y_train.values, feature_names=X_train.columns.tolist())
    modelo.fit_pseudoinverse()

    pred_train = modelo.predict(X_train.values)
    pred_validation = modelo.predict(X_validation.values)

    resumen = {
        "modelo": nombre_modelo,
        "escenario_target": escenario_target,
        **_resumir_metricas_train_y_validation(y_train, pred_train, y_validation, pred_validation),
    }

    metricas_por_zona = None
    if zonas_validation is not None:
        tabla = []
        zonas_validation = pd.Series(zonas_validation).reset_index(drop=True)
        y_val_reset = pd.Series(y_validation).reset_index(drop=True)
        pred_val_reset = pd.Series(pred_validation)

        for zona in zonas_validation.unique():
            mascara = zonas_validation == zona
            tabla.append(
                {
                    "modelo": nombre_modelo,
                    "escenario_target": escenario_target,
                    "zona": zona,
                    "MAE_validation": mae(y_val_reset[mascara], pred_val_reset[mascara]),
                    "RMSE_validation": rmse(y_val_reset[mascara], pred_val_reset[mascara]),
                    "R2_validation": r2(y_val_reset[mascara], pred_val_reset[mascara]),
                }
            )
        metricas_por_zona = pd.DataFrame(tabla)

    return modelo, pred_train, pred_validation, resumen, metricas_por_zona


def correlacion_segura(x, y):
    """Calcula correlacion solo si ambas series tienen variacion real."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def construir_ranking_features_candidatas(
    dataframe_train,
    columnas_base=None,
    columnas_excluir=None,
    columna_target="precio",
):
    columnas_base = set(columnas_base or [])
    columnas_excluir = set(columnas_excluir or [])

    filas = []
    for columna in dataframe_train.columns:
        if columna == columna_target or columna in columnas_base or columna in columnas_excluir:
            continue
        serie = dataframe_train[columna]
        if not pd.api.types.is_numeric_dtype(serie):
            continue
        serie = serie.astype(float)

        if serie.notna().sum() != len(serie):
            continue
        if serie.std(ddof=0) < 1e-9:
            continue

        corr = correlacion_segura(serie.values, dataframe_train[columna_target].values)
        if np.isnan(corr):
            continue
        filas.append(
            {
                "feature": columna,
                "corr_con_precio": corr,
                "abs_corr_con_precio": abs(corr),
            }
        )

    return pd.DataFrame(filas).sort_values("abs_corr_con_precio", ascending=False).reset_index(drop=True)

def evaluar_curva_complejidad_features(
    df_train_modelo,
    df_validation_modelo,
    columnas_base,
    columnas_extra_ordenadas,
    columnas_categoricas=None,
    puntos_extra=None,
    nombre_modelo_base="M4",
):
    if puntos_extra is None:
        puntos_extra = [0, 4, 8, 12, 16, 20, 30, 40, 50, 60]
    else:
        puntos_extra = list(dict.fromkeys(int(x) for x in puntos_extra))

    columnas_categoricas = columnas_categoricas or []
    filas = []
    for cantidad_extra in puntos_extra:
        columnas_numericas = columnas_base + columnas_extra_ordenadas[:cantidad_extra]
        prep = preparar_features_para_modelo(
            df_train_modelo,
            df_validation_modelo,
            columnas_numericas,
            columnas_categoricas,
        )
        _, _, _, resumen, _ = entrenar_y_resumir_modelo(
            f"{nombre_modelo_base}_plus_{cantidad_extra}",
            "precio_observado",
            prep["X_train"],
            df_train_modelo["precio"],
            prep["X_validation"],
            df_validation_modelo["precio"],
            df_validation_modelo["zona"],
        )
        filas.append(
            {
                "cantidad_extras": cantidad_extra,
                "cantidad_features_numericas": len(columnas_numericas),
                "cantidad_features_diseno": prep["X_train"].shape[1],
                "RMSE_train": resumen["RMSE_train"],
                "RMSE_validation": resumen["RMSE_validation"],
                "gap_RMSE_validation_menos_train": resumen["RMSE_validation"] - resumen["RMSE_train"],
            }
        )

    return pd.DataFrame(filas)


def entrenar_modelo_regularizado_y_evaluar(
    X_train,
    y_train,
    X_validation,
    y_validation,
    feature_names,
    tipo_regularizacion="ridge",
    lambda_reg=0.0,
    learning_rate=0.01,
    max_iteraciones=30000,
    tolerancia=1e-8,
    pesos_iniciales=None,
):
    if tipo_regularizacion == "ridge":
        modelo = RegresionLineal(X_train, y_train, feature_names=feature_names, l2=lambda_reg, l1=0.0)
        modelo.fit_pseudoinverse()
    elif tipo_regularizacion == "lasso":
        modelo = RegresionLineal(X_train, y_train, feature_names=feature_names, l2=0.0, l1=lambda_reg)
        modelo.fit_gradient_descent(
            learning_rate=learning_rate,
            n_iterations=max_iteraciones,
            tolerance=tolerancia,
            initial_weights=pesos_iniciales,
        )
    else:
        raise ValueError("tipo_regularizacion debe ser 'ridge' o 'lasso'.")

    pred_train = modelo.predict(X_train)
    pred_validation = modelo.predict(X_validation)

    resumen = {
        "tipo_regularizacion": tipo_regularizacion,
        "lambda": float(lambda_reg),
        **_resumir_metricas_train_y_validation(y_train, pred_train, y_validation, pred_validation),
        "cantidad_coeficientes_casi_cero": modelo.contar_coeficientes_casi_cero(tolerancia=1e-4),
    }
    return modelo, resumen


def construir_regularization_path(
    X_train,
    y_train,
    feature_names,
    lambdas,
    tipo_regularizacion="ridge",
    learning_rate=0.01,
    max_iteraciones=30000,
    tolerancia=1e-8,
    warm_start=False,
):
    filas = []
    pesos_previos = None

    for lambda_reg in sorted(lambdas):
        modelo, _ = entrenar_modelo_regularizado_y_evaluar(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_train,
            y_validation=y_train,
            feature_names=feature_names,
            tipo_regularizacion=tipo_regularizacion,
            lambda_reg=lambda_reg,
            learning_rate=learning_rate,
            max_iteraciones=max_iteraciones,
            tolerancia=tolerancia,
            pesos_iniciales=pesos_previos if warm_start else None,
        )

        if warm_start:
            pesos_previos = modelo.coef.copy()

        coef_dict = modelo.get_coefficient_dict()
        for feature, coeficiente in coef_dict.items():
            filas.append(
                {
                    "lambda": float(lambda_reg),
                    "feature": feature,
                    "coeficiente": coeficiente,
                    "tipo_regularizacion": tipo_regularizacion,
                }
            )

    return pd.DataFrame(filas)


def cross_validate_lambdas_con_preprocesamiento(
    dataframe_modelo,
    columnas_numericas,
    columnas_categoricas,
    lambdas,
    columna_target="precio",
    tipo_regularizacion="ridge",
    n_splits=5,
    seed=42,
    learning_rate=0.01,
    max_iteraciones=30000,
    tolerancia=1e-8,
):
    folds = generar_folds_kfold(len(dataframe_modelo), n_splits=n_splits, shuffle=True, seed=seed)
    resultados = []

    for lambda_reg in sorted(lambdas):
        ecm_folds = []
        cantidad_ceros = []

        for train_idx, val_idx in folds:
            df_fold_train = dataframe_modelo.iloc[train_idx].reset_index(drop=True)
            df_fold_val = dataframe_modelo.iloc[val_idx].reset_index(drop=True)

            prep = preparar_features_para_modelo(
                df_fold_train,
                df_fold_val,
                columnas_numericas,
                columnas_categoricas,
            )

            feature_names = prep["X_train"].columns.tolist()
            modelo, _ = entrenar_modelo_regularizado_y_evaluar(
                X_train=prep["X_train"].values,
                y_train=df_fold_train[columna_target].values,
                X_validation=prep["X_validation"].values,
                y_validation=df_fold_val[columna_target].values,
                feature_names=feature_names,
                tipo_regularizacion=tipo_regularizacion,
                lambda_reg=lambda_reg,
                learning_rate=learning_rate,
                max_iteraciones=max_iteraciones,
                tolerancia=tolerancia,
            )

            pred_val = modelo.predict(prep["X_validation"].values)
            ecm_folds.append(mse(df_fold_val[columna_target].values, pred_val))
            cantidad_ceros.append(modelo.contar_coeficientes_casi_cero(tolerancia=1e-4))

        resultados.append(
            {
                "tipo_regularizacion": tipo_regularizacion,
                "lambda": float(lambda_reg),
                "ECM_promedio_cv": float(np.mean(ecm_folds)),
                "ECM_std_cv": float(np.std(ecm_folds)),
                "coeficientes_casi_cero_promedio": float(np.mean(cantidad_ceros)),
            }
        )

    return pd.DataFrame(resultados)


def evaluar_metricas_regularizacion_en_grilla(
    X_train,
    y_train,
    X_validation,
    y_validation,
    feature_names,
    lambdas,
    tipo_regularizacion="ridge",
    learning_rate=0.01,
    max_iteraciones=30000,
    tolerancia=1e-8,
):
    filas = []

    for lambda_reg in sorted(lambdas):
        _, resumen = entrenar_modelo_regularizado_y_evaluar(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            feature_names=feature_names,
            tipo_regularizacion=tipo_regularizacion,
            lambda_reg=lambda_reg,
            learning_rate=learning_rate,
            max_iteraciones=max_iteraciones,
            tolerancia=tolerancia,
        )
        filas.append(
            {
                "tipo_regularizacion": tipo_regularizacion,
                "lambda": float(lambda_reg),
                "ECM_train": resumen["ECM_train"],
                "MAE_train": resumen["MAE_train"],
                "RMSE_train": resumen["RMSE_train"],
                "R2_train": resumen["R2_train"],
                "ECM_validation": resumen["ECM_validation"],
                "MAE_validation": resumen["MAE_validation"],
                "RMSE_validation": resumen["RMSE_validation"],
                "R2_validation": resumen["R2_validation"],
                "cantidad_coeficientes_casi_cero": resumen["cantidad_coeficientes_casi_cero"],
                "gap_RMSE_validation_menos_train": resumen["RMSE_validation"] - resumen["RMSE_train"],
            }
        )

    return pd.DataFrame(filas)


def construir_tablas_coeficientes_extremos(tabla_coeficientes, n_extremos=10):
    tabla_sin_intercepto = tabla_coeficientes[tabla_coeficientes["feature"] != "intercepto"].reset_index(drop=True)
    n_extremos = min(n_extremos, max(1, len(tabla_sin_intercepto) // 2))

    tabla_mayores = tabla_sin_intercepto.head(n_extremos).reset_index(drop=True)
    tabla_menores = tabla_sin_intercepto.tail(n_extremos).reset_index(drop=True)
    return tabla_mayores, tabla_menores


def ajustar_modelo_con_preprocesamiento(
    df_train_modelo,
    df_eval_modelo,
    columnas_numericas,
    columnas_categoricas,
    columna_target="precio",
    tipo_modelo="ols",
    lambda_reg=0.0,
    learning_rate=0.01,
    max_iteraciones=30000,
    tolerancia=1e-8,
):
    """Entrena sobre train y evalua en train y eval."""
    columnas_categoricas = columnas_categoricas or []

    prep = preparar_features_para_modelo(
        df_train_modelo,
        df_eval_modelo,
        columnas_numericas,
        columnas_categoricas,
    )

    X_train = prep["X_train"].values
    y_train = df_train_modelo[columna_target].values
    X_eval = prep["X_validation"].values
    y_eval = df_eval_modelo[columna_target].values
    feature_names = prep["X_train"].columns.tolist()

    if tipo_modelo == "ols":
        modelo = RegresionLineal(X_train, y_train, feature_names=feature_names)
        modelo.fit_pseudoinverse()
    elif tipo_modelo in {"ridge", "lasso"}:
        modelo, _ = entrenar_modelo_regularizado_y_evaluar(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_eval,
            y_validation=y_eval,
            feature_names=feature_names,
            tipo_regularizacion=tipo_modelo,
            lambda_reg=lambda_reg,
            learning_rate=learning_rate,
            max_iteraciones=max_iteraciones,
            tolerancia=tolerancia,
        )
    else:
        raise ValueError("tipo_modelo debe ser 'ols', 'ridge' o 'lasso'.")

    pred_train = modelo.predict(X_train)
    pred_eval = modelo.predict(X_eval)

    return {
        "modelo_entrenado": modelo,
        "preparacion": prep,
        "pred_train": pred_train,
        "pred_eval": pred_eval,
        "metricas_train": resumir_metricas_regresion(y_train, pred_train),
        "metricas_eval": resumir_metricas_regresion(y_eval, pred_eval),
    }


def calcular_learning_curve_modelo(
    nombre_modelo,
    df_train_modelo,
    df_validation_modelo,
    columnas_numericas,
    columnas_categoricas,
    columna_target="precio",
    tipo_modelo="ols",
    lambda_reg=0.0,
    learning_rate=0.01,
    max_iteraciones=30000,
    tolerancia=1e-8,
    fracciones=None,
    seed=42,
):
    """Arma una learning curve con validation fijo."""
    if fracciones is None:
        fracciones = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]

    rng = np.random.default_rng(seed)
    indices_permutados = rng.permutation(len(df_train_modelo))
    minimo_observaciones = max(30, len(columnas_numericas) + max(1, len(columnas_categoricas)) + 5)

    filas = []
    tamanios_vistos = set()

    for fraccion in fracciones:
        cantidad = int(round(fraccion * len(df_train_modelo)))
        cantidad = max(minimo_observaciones, cantidad)
        cantidad = min(len(df_train_modelo), cantidad)
        if cantidad in tamanios_vistos:
            continue
        tamanios_vistos.add(cantidad)

        subset_idx = np.sort(indices_permutados[:cantidad])
        df_subset = df_train_modelo.iloc[subset_idx].reset_index(drop=True)

        resultado = ajustar_modelo_con_preprocesamiento(
            df_train_modelo=df_subset,
            df_eval_modelo=df_validation_modelo,
            columnas_numericas=columnas_numericas,
            columnas_categoricas=columnas_categoricas,
            columna_target=columna_target,
            tipo_modelo=tipo_modelo,
            lambda_reg=lambda_reg,
            learning_rate=learning_rate,
            max_iteraciones=max_iteraciones,
            tolerancia=tolerancia,
        )

        filas.append(
            {
                "modelo": nombre_modelo,
                "n_train": cantidad,
                "fraccion_train": cantidad / len(df_train_modelo),
                "MAE_train": resultado["metricas_train"]["MAE"],
                "RMSE_train": resultado["metricas_train"]["RMSE"],
                "R2_train": resultado["metricas_train"]["R2"],
                "MAE_validation": resultado["metricas_eval"]["MAE"],
                "RMSE_validation": resultado["metricas_eval"]["RMSE"],
                "R2_validation": resultado["metricas_eval"]["R2"],
            }
        )

    return pd.DataFrame(filas)


def exportar_csv_inspeccion_modelos(output_path, splits):
    """Exporta un CSV con columnas originales y bloques auxiliares."""
    tablas = []
    for split in splits:
        base = split["dataframe_original"].reset_index(drop=True).copy()
        partes = [pd.DataFrame({"split": [split["split_name"]] * len(base)}), base]
        columnas_originales = set(base.columns)

        for nombre_modelo, info_modelo in split["modelos"].items():
            engineered_df = info_modelo.get("engineered_df")
            if engineered_df is not None:
                engineered_df = engineered_df.reset_index(drop=True)
                columnas_nuevas = [col for col in engineered_df.columns if col not in columnas_originales]
                if columnas_nuevas:
                    partes.append(engineered_df[columnas_nuevas].add_prefix(f"{nombre_modelo}__feat__"))

            for clave, prefijo in [
                ("scaled_df", f"{nombre_modelo}__scaled__"),
                ("dummy_df", f"{nombre_modelo}__dummy__"),
                ("design_df", f"{nombre_modelo}__model__"),
            ]:
                bloque = info_modelo.get(clave)
                if bloque is not None and not bloque.empty:
                    partes.append(bloque.reset_index(drop=True).add_prefix(prefijo))

        tablas.append(pd.concat(partes, axis=1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tabla_final = pd.concat(tablas, axis=0, ignore_index=True)
    tabla_final.to_csv(output_path, index=False)
    return tabla_final
