"""Preprocesamiento y feature engineering."""

import pandas as pd
import numpy as np

CONVERSION_SQFT_A_M2 = 0.092903


def _validar_columnas_requeridas(dataframe, columnas_requeridas, contexto):
    faltantes = [col for col in columnas_requeridas if col not in dataframe.columns]
    if faltantes:
        columnas_str = ", ".join(str(col) for col in faltantes)
        raise ValueError(f"{contexto} requiere columnas que no estan presentes: {columnas_str}.")


def agregar_columna_zona(dataframe):
    """Crea la columna `zona` a partir de la latitud."""
    copia = dataframe.copy()
    _validar_columnas_requeridas(copia, ["lat"], "agregar_columna_zona")
    copia["zona"] = np.where(copia["lat"] >= 0, "new_york", "buenos_aires")
    return copia


def convertir_superficies_a_m2(dataframe):
    """Pasa superficies a m2 cuando vienen en sqft."""
    copia = dataframe.copy()
    _validar_columnas_requeridas(copia, ["unidades", "Área", "metros_cubiertos"], "convertir_superficies_a_m2")
    mascara_sqft = copia["unidades"] == "sqft"
    copia.loc[mascara_sqft, "Área"] = copia.loc[mascara_sqft, "Área"] * CONVERSION_SQFT_A_M2
    copia.loc[mascara_sqft, "metros_cubiertos"] = (
        copia.loc[mascara_sqft, "metros_cubiertos"] * CONVERSION_SQFT_A_M2
    )
    copia["unidades"] = "m2"
    return copia


def eliminar_filas_sin_precio(dataframe):
    """Elimina filas en las que falta el target `precio`."""
    copia = dataframe.copy()
    _validar_columnas_requeridas(copia, ["precio"], "eliminar_filas_sin_precio")
    return copia.dropna(subset=["precio"])


def filtrar_precios_validos(dataframe, precio_minimo=1):
    """Conserva filas con `precio` mayor o igual al umbral indicado."""
    copia = dataframe.copy()
    _validar_columnas_requeridas(copia, ["precio"], "filtrar_precios_validos")
    return copia[copia["precio"] >= precio_minimo]


def agregar_features_basicas(dataframe):
    """Agrega algunas variables base."""
    copia = dataframe.copy()
    _validar_columnas_requeridas(copia, ["Área", "metros_cubiertos"], "agregar_features_basicas")
    copia["superficie_descubierta"] = (copia["Área"] - copia["metros_cubiertos"]).clip(lower=0)
    copia["proporcion_cubierta"] = np.where(
        copia["Área"] > 0,
        copia["metros_cubiertos"] / copia["Área"],
        np.nan,
    )
    return copia


def construir_dataset_base(dataframe, precio_minimo_venta=None):
    """Deja listo el dataset base."""
    copia = dataframe.copy()
    copia = agregar_columna_zona(copia)
    copia = convertir_superficies_a_m2(copia)
    copia = eliminar_filas_sin_precio(copia)
    copia = filtrar_precios_validos(copia, precio_minimo=1)
    if precio_minimo_venta is not None:
        copia = filtrar_precios_validos(copia, precio_minimo=precio_minimo_venta)
    copia = agregar_features_basicas(copia)
    return copia


def aprender_mediana_pisos_por_tipo(df_train):
    """Aprende la mediana de `pisos` por tipo usando solo train."""
    _validar_columnas_requeridas(df_train, ["tipo", "pisos"], "aprender_mediana_pisos_por_tipo")
    mediana_tipo = df_train.groupby("tipo")["pisos"].median()
    mediana_global = df_train["pisos"].median()
    if pd.isna(mediana_global):
        mediana_global = 1
    mediana_tipo = mediana_tipo.fillna(mediana_global).round()
    return mediana_tipo.to_dict()


def imputar_pisos_faltantes(dataframe, mediana_pisos_por_tipo):
    """Completa `pisos` faltantes con la mediana correspondiente al tipo."""
    copia = dataframe.copy()
    _validar_columnas_requeridas(copia, ["tipo", "pisos"], "imputar_pisos_faltantes")
    valores_validos = [v for v in mediana_pisos_por_tipo.values() if pd.notna(v)]
    valor_respaldo = int(round(np.median(valores_validos))) if valores_validos else 1

    def completar(fila):
        if pd.notna(fila["pisos"]):
            return fila["pisos"]
        tipo = fila["tipo"]
        if tipo in mediana_pisos_por_tipo and pd.notna(mediana_pisos_por_tipo[tipo]):
            return int(round(mediana_pisos_por_tipo[tipo]))
        return valor_respaldo

    copia["pisos"] = copia.apply(completar, axis=1)
    return copia


def aprender_mediana_edad_por_tipo(df_train):
    """Aprende la mediana de `edad` por tipo usando solo train."""
    _validar_columnas_requeridas(df_train, ["tipo", "edad"], "aprender_mediana_edad_por_tipo")
    return df_train.groupby("tipo")["edad"].median().to_dict()


def imputar_edad_faltante(dataframe, mediana_edad_por_tipo):
    """Completa `edad` faltante con la mediana correspondiente al tipo."""
    copia = dataframe.copy()
    _validar_columnas_requeridas(copia, ["tipo", "edad"], "imputar_edad_faltante")
    valores_validos = [v for v in mediana_edad_por_tipo.values() if pd.notna(v)]
    valor_respaldo = float(np.median(valores_validos)) if valores_validos else 10.0

    def completar(fila):
        if pd.notna(fila["edad"]):
            return fila["edad"]
        tipo = fila["tipo"]
        if tipo in mediana_edad_por_tipo and pd.notna(mediana_edad_por_tipo[tipo]):
            return float(mediana_edad_por_tipo[tipo])
        return valor_respaldo

    copia["edad"] = copia.apply(completar, axis=1)
    return copia


def agregar_categoria_edad(dataframe):
    """
    Crea una variable categórica simple a partir de `edad`.
    """
    copia = dataframe.copy()
    _validar_columnas_requeridas(copia, ["edad"], "agregar_categoria_edad")
    bins = [-np.inf, 10, 25, np.inf]
    labels = ["nueva", "media", "antigua"]
    copia["categoria_edad"] = pd.cut(copia["edad"], bins=bins, labels=labels, right=False)
    return copia


def aplicar_imputacion_basica(dataframe, mediana_pisos_por_tipo, mediana_edad_por_tipo):
    """Aplica imputación de pisos y edad, y luego agrega `categoria_edad`."""
    copia = dataframe.copy()
    copia = imputar_pisos_faltantes(copia, mediana_pisos_por_tipo)
    copia = imputar_edad_faltante(copia, mediana_edad_por_tipo)
    copia = agregar_categoria_edad(copia)
    return copia


def _division_segura(numerador, denominador):
    """Divide evitando denominadores nulos o no positivos."""
    numerador = pd.Series(numerador).astype(float)
    denominador = pd.Series(denominador).astype(float)
    return pd.Series(
        np.where(denominador.notna() & (denominador > 0), numerador / denominador, np.nan),
        index=numerador.index,
    )


def _division_segura_no_cero(numerador, denominador):
    """Divide evitando denominadores nulos, aunque sean negativos."""
    numerador = pd.Series(numerador).astype(float)
    denominador = pd.Series(denominador).astype(float)
    return pd.Series(
        np.where(denominador.notna() & (denominador != 0), numerador / denominador, np.nan),
        index=numerador.index,
    )


def _construir_indicador_binario_pileta(serie_pileta):
    """Pasa `pileta` a 0/1."""
    serie = pd.Series(serie_pileta).astype(str).str.strip().str.lower()
    positivos = {"1", "true", "t", "yes", "y", "si", "sí", "con pileta"}
    return serie.isin(positivos).astype(int)


COLUMNAS_CATEGORICAS_MODELOS = ["tipo", "zona"]

# Listas finales de features para M4 y M5.
COLUMNAS_NUMERICAS_M4 = [
    "Área",
    "metros_cubiertos",
    "ambientes",
    "edad",
    "superficie_descubierta",
    "proporcion_cubierta",
    "area_por_ambiente",
    "cubiertos_por_ambiente",
    "area2",
    "metros_cubiertos2",
    "edad2",
    "lat",
    "lon",
    "lat2",
    "lon2",
    "lat_x_lon",
    "area_x_zona_new_york",
    "metros_x_zona_new_york",
    "ambientes_x_zona_new_york",
    "edad_x_zona_new_york",
    "area_x_tipo_depto",
    "area_x_tipo_ph",
]

COLUMNAS_NUMERICAS_M5 = [
    "Área",
    "metros_cubiertos",
    "ambientes",
    "edad",
    "superficie_descubierta",
    "proporcion_cubierta",
    "area_por_ambiente",
    "cubiertos_por_ambiente",
    "area2",
    "metros_cubiertos2",
    "edad2",
    "area_x_zona_new_york",
    "metros_x_zona_new_york",
    "ambientes_x_zona_new_york",
    "edad_x_zona_new_york",
    "edad_x_lon",
    "area_x_tipo_depto",
    "area_x_tipo_ph",
    "metros_x_tipo_depto",
    "metros_x_tipo_ph",
    "lat",
    "lon",
    "lat2",
    "lon2",
    "lat_x_lon",
    "area_x_lat",
    "area_x_lon",
    "metros_x_lat",
    "metros_x_lon",
    "lat_log1p",
    "lat_div_lon",
    "lon_x_indicador_zona_new_york",
    "superficie_descubierta_log1p",
    "lat_x_indicador_zona_new_york",
    "edad_cube",
    "pisos",
    "lon_div_lat",
    "ambientes_x_edad",
    "area_por_ambiente_div_edad",
    "superficie_descubierta_x_area_por_ambiente",
    "proporcion_cubierta_div_ambientes",
    "area_x_area_por_ambiente",
    "area_por_ambiente_x_cubiertos_por_ambiente",
    "ambientes_log1p",
    "area_por_ambiente_sqrt",
    "ambientes_x_superficie_descubierta",
    "metros_cubiertos_div_lon",
    "area_div_lon",
    "lat_div_cubiertos_por_ambiente",
    "area_x_pisos",
]


def agregar_features_derivadas_m4(dataframe):
    """Agrega las features de M4."""
    copia = dataframe.copy()
    _validar_columnas_requeridas(
        copia,
        ["zona", "tipo", "pileta", "Área", "metros_cubiertos", "ambientes", "edad", "pisos", "lat", "lon"],
        "agregar_features_derivadas_m4",
    )
    if "superficie_descubierta" not in copia.columns or "proporcion_cubierta" not in copia.columns:
        copia = agregar_features_basicas(copia)

    copia["lat2"] = copia["lat"] ** 2
    copia["lon2"] = copia["lon"] ** 2
    copia["lat_x_lon"] = copia["lat"] * copia["lon"]
    copia["area_x_lat"] = copia["Área"] * copia["lat"]
    copia["area_x_lon"] = copia["Área"] * copia["lon"]
    copia["metros_x_lat"] = copia["metros_cubiertos"] * copia["lat"]
    copia["metros_x_lon"] = copia["metros_cubiertos"] * copia["lon"]

    copia["indicador_zona_new_york"] = (copia["zona"] == "new_york").astype(int)
    copia["indicador_tipo_depto"] = (copia["tipo"] == "depto").astype(int)
    copia["indicador_tipo_ph"] = (copia["tipo"] == "ph").astype(int)
    copia["indicador_pileta"] = _construir_indicador_binario_pileta(copia["pileta"])

    copia["area_por_ambiente"] = _division_segura(copia["Área"], copia["ambientes"])
    copia["cubiertos_por_ambiente"] = _division_segura(copia["metros_cubiertos"], copia["ambientes"])

    copia["area2"] = copia["Área"] ** 2
    copia["metros_cubiertos2"] = copia["metros_cubiertos"] ** 2
    copia["edad2"] = copia["edad"] ** 2

    copia["area_x_zona_new_york"] = copia["Área"] * copia["indicador_zona_new_york"]
    copia["metros_x_zona_new_york"] = copia["metros_cubiertos"] * copia["indicador_zona_new_york"]
    copia["ambientes_x_zona_new_york"] = copia["ambientes"] * copia["indicador_zona_new_york"]
    copia["edad_x_zona_new_york"] = copia["edad"] * copia["indicador_zona_new_york"]
    copia["edad_x_lon"] = copia["edad"] * copia["lon"]

    copia["area_x_tipo_depto"] = copia["Área"] * copia["indicador_tipo_depto"]
    copia["area_x_tipo_ph"] = copia["Área"] * copia["indicador_tipo_ph"]
    copia["metros_x_tipo_depto"] = copia["metros_cubiertos"] * copia["indicador_tipo_depto"]
    copia["metros_x_tipo_ph"] = copia["metros_cubiertos"] * copia["indicador_tipo_ph"]
    copia["lat_log1p"] = np.log1p(np.clip(copia["lat"], a_min=0, a_max=None))
    copia["lat_div_lon"] = _division_segura_no_cero(copia["lat"], copia["lon"])
    copia["lon_x_indicador_zona_new_york"] = copia["lon"] * copia["indicador_zona_new_york"]
    copia["superficie_descubierta_log1p"] = np.log1p(np.clip(copia["superficie_descubierta"], a_min=0, a_max=None))
    copia["lat_x_indicador_zona_new_york"] = copia["lat"] * copia["indicador_zona_new_york"]
    copia["edad_cube"] = copia["edad"] ** 3
    copia["lon_div_lat"] = _division_segura_no_cero(copia["lon"], copia["lat"])
    copia["ambientes_x_edad"] = copia["ambientes"] * copia["edad"]
    copia["area_por_ambiente_div_edad"] = _division_segura(copia["area_por_ambiente"], copia["edad"])
    copia["superficie_descubierta_x_area_por_ambiente"] = (
        copia["superficie_descubierta"] * copia["area_por_ambiente"]
    )
    copia["proporcion_cubierta_div_ambientes"] = _division_segura(copia["proporcion_cubierta"], copia["ambientes"])
    copia["area_x_area_por_ambiente"] = copia["Área"] * copia["area_por_ambiente"]
    copia["area_por_ambiente_x_cubiertos_por_ambiente"] = (
        copia["area_por_ambiente"] * copia["cubiertos_por_ambiente"]
    )
    copia["ambientes_log1p"] = np.log1p(np.clip(copia["ambientes"], a_min=0, a_max=None))
    copia["area_por_ambiente_sqrt"] = np.sqrt(np.clip(copia["area_por_ambiente"], a_min=0, a_max=None))
    copia["ambientes_x_superficie_descubierta"] = copia["ambientes"] * copia["superficie_descubierta"]
    return copia


def obtener_configuracion_m4():
    """Devuelve las columnas de M4."""
    return COLUMNAS_NUMERICAS_M4.copy(), COLUMNAS_CATEGORICAS_MODELOS.copy()


def agregar_features_derivadas_m5(dataframe):
    """Agrega las features extra de M5."""
    copia = agregar_features_derivadas_m4(dataframe.copy())

    copia["metros_cubiertos_div_lon"] = _division_segura_no_cero(copia["metros_cubiertos"], copia["lon"])
    copia["area_div_lon"] = _division_segura_no_cero(copia["Área"], copia["lon"])
    copia["lat_div_cubiertos_por_ambiente"] = _division_segura_no_cero(copia["lat"], copia["cubiertos_por_ambiente"])
    copia["area_x_pisos"] = copia["Área"] * copia["pisos"]
    return copia


def obtener_configuracion_m5():
    """Devuelve las columnas de M5."""
    return COLUMNAS_NUMERICAS_M5.copy(), COLUMNAS_CATEGORICAS_MODELOS.copy()


def agregar_features_candidatas_combinacionales(dataframe):
    """Arma un pool grande de candidatas a partir de M5."""
    copia = agregar_features_derivadas_m5(dataframe.copy())
    area_col = "Área"
    base_cont = [
        area_col,
        "metros_cubiertos",
        "ambientes",
        "edad",
        "pisos",
        "superficie_descubierta",
        "proporcion_cubierta",
        "lat",
        "lon",
        "area_por_ambiente",
        "cubiertos_por_ambiente",
    ]
    base_cont = [col for col in base_cont if col in copia.columns]
    indicadores = [
        col
        for col in ["indicador_zona_new_york", "indicador_tipo_depto", "indicador_tipo_ph", "indicador_pileta"]
        if col in copia.columns
    ]

    nuevas = {}

    for col in base_cont:
        serie = copia[col].astype(float)
        nuevas[f"{col}_log1p_pool"] = np.log1p(np.clip(serie, a_min=0, a_max=None))
        nuevas[f"{col}_sqrt_pool"] = np.sqrt(np.clip(serie, a_min=0, a_max=None))

    for col in [area_col, "metros_cubiertos", "edad", "superficie_descubierta", "ambientes", "pisos"]:
        if col in copia.columns:
            nuevas[f"{col}_cube_pool"] = copia[col].astype(float) ** 3

    for i, col_a in enumerate(base_cont):
        for col_b in base_cont[i + 1:]:
            nuevas[f"{col_a}_x_{col_b}_pool"] = copia[col_a].astype(float) * copia[col_b].astype(float)

    for col_a in base_cont:
        for col_b in base_cont:
            if col_a == col_b:
                continue
            nuevas[f"{col_a}_div_{col_b}_pool"] = _division_segura_no_cero(copia[col_a], copia[col_b])

    for col in base_cont:
        for indicador in indicadores:
            nuevas[f"{col}_x_{indicador}_pool"] = copia[col].astype(float) * copia[indicador].astype(float)

    return pd.concat([copia.reset_index(drop=True), pd.DataFrame(nuevas, index=copia.index).reset_index(drop=True)], axis=1)


def preparar_matrices_para_modelado(
    train_df,
    validation_df,
    test_df,
    columnas_numericas,
    columnas_categoricas,
    columna_target="precio",
):
    """Prepara matrices para train, validation y test."""
    _validar_columnas_requeridas(train_df, [columna_target], "preparar_matrices_para_modelado(train_df)")
    _validar_columnas_requeridas(validation_df, [columna_target], "preparar_matrices_para_modelado(validation_df)")
    _validar_columnas_requeridas(test_df, [columna_target], "preparar_matrices_para_modelado(test_df)")
    _validar_columnas_requeridas(train_df, columnas_numericas, "preparar_matrices_para_modelado(train_df)")
    _validar_columnas_requeridas(validation_df, columnas_numericas, "preparar_matrices_para_modelado(validation_df)")
    _validar_columnas_requeridas(test_df, columnas_numericas, "preparar_matrices_para_modelado(test_df)")

    X_train_numerico = train_df[columnas_numericas].copy()
    X_validation_numerico = validation_df[columnas_numericas].copy()
    X_test_numerico = test_df[columnas_numericas].copy()

    medias_train = X_train_numerico.mean(axis=0)
    desvios_train = X_train_numerico.std(axis=0, ddof=0).replace(0, 1)
    X_train_numerico_escalado = (X_train_numerico - medias_train) / desvios_train
    X_validation_numerico_escalado = (X_validation_numerico - medias_train) / desvios_train
    X_test_numerico_escalado = (X_test_numerico - medias_train) / desvios_train

    if columnas_categoricas:
        _validar_columnas_requeridas(train_df, columnas_categoricas, "preparar_matrices_para_modelado(train_df)")
        _validar_columnas_requeridas(validation_df, columnas_categoricas, "preparar_matrices_para_modelado(validation_df)")
        _validar_columnas_requeridas(test_df, columnas_categoricas, "preparar_matrices_para_modelado(test_df)")
        columnas_dummy = pd.get_dummies(train_df[columnas_categoricas], drop_first=True, dtype=int).columns.tolist()

        def alinear_dummies(dataframe):
            dummies = pd.get_dummies(dataframe[columnas_categoricas], drop_first=True, dtype=int)
            for columna in columnas_dummy:
                if columna not in dummies.columns:
                    dummies[columna] = 0
            columnas_extra = [columna for columna in dummies.columns if columna not in columnas_dummy]
            if columnas_extra:
                dummies = dummies.drop(columns=columnas_extra)
            return dummies[columnas_dummy]

        X_train_categorico = alinear_dummies(train_df)
        X_validation_categorico = alinear_dummies(validation_df)
        X_test_categorico = alinear_dummies(test_df)
    else:
        columnas_dummy = []
        X_train_categorico = pd.DataFrame(index=train_df.index)
        X_validation_categorico = pd.DataFrame(index=validation_df.index)
        X_test_categorico = pd.DataFrame(index=test_df.index)

    X_train = pd.concat([X_train_numerico_escalado.reset_index(drop=True), X_train_categorico.reset_index(drop=True)], axis=1)
    X_validation = pd.concat(
        [X_validation_numerico_escalado.reset_index(drop=True), X_validation_categorico.reset_index(drop=True)],
        axis=1,
    )
    X_test = pd.concat([X_test_numerico_escalado.reset_index(drop=True), X_test_categorico.reset_index(drop=True)], axis=1)

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": train_df[columna_target].copy(),
        "y_validation": validation_df[columna_target].copy(),
        "y_test": test_df[columna_target].copy(),
        "X_train_numerico": X_train_numerico,
        "X_validation_numerico": X_validation_numerico,
        "X_test_numerico": X_test_numerico,
        "X_train_numerico_escalado": X_train_numerico_escalado,
        "X_validation_numerico_escalado": X_validation_numerico_escalado,
        "X_test_numerico_escalado": X_test_numerico_escalado,
        "X_train_categorico": X_train_categorico,
        "X_validation_categorico": X_validation_categorico,
        "X_test_categorico": X_test_categorico,
        "medias_train": medias_train,
        "desvios_train": desvios_train,
        "columnas_dummy": columnas_dummy,
    }
