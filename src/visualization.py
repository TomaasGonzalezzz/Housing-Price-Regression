"""Funciones de visualizacion del notebook."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


COLORES_ZONA = {
    "buenos_aires": "#9EC5E5",
    "new_york": "#F6C27A",
}

COLORES_MODELO = {
    "M1": "#A7C7E7",
    "M2": "#F9D29D",
    "M3": "#A8D5BA",
    "M4": "#C7B8EA",
    "M5": "#F4A7B9",
    "Ridge": "#7FB3D5",
    "Lasso": "#E59866",
    "M6": "#76D7C4",
}


def _validar_columnas_requeridas(dataframe, columnas_requeridas, contexto):
    faltantes = [col for col in columnas_requeridas if col not in dataframe.columns]
    if faltantes:
        columnas_str = ", ".join(str(col) for col in faltantes)
        raise ValueError(f"{contexto} requiere columnas que no estan presentes: {columnas_str}.")


def configurar_estilo_graficos():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "axes.facecolor": "#FCFCFC",
        "figure.facecolor": "white",
    })


def _log_seguro(valores):
    valores = np.asarray(valores, dtype=float)
    return np.log1p(np.clip(valores, a_min=0, a_max=None))


def _moneda(x, pos):
    return f"{x:,.0f}".replace(",", ".")


def _formatear_numero_entero(valor):
    return f"{valor:,.0f}".replace(",", ".")


def _anotar_barras(ax, barras, desfase_relativo=0.02, fontsize=8):
    alturas = [barra.get_height() for barra in barras]
    if len(alturas) == 0:
        return

    altura_maxima = max(alturas)
    desplazamiento = altura_maxima * desfase_relativo if altura_maxima > 0 else 0.0

    for barra in barras:
        altura = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura + desplazamiento,
            _formatear_numero_entero(altura),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )


def _porcentajes_en_bandas(y_real, y_predicho, tolerancias=(0.10, 0.25)):
    y_real = np.asarray(y_real, dtype=float).reshape(-1)
    y_predicho = np.asarray(y_predicho, dtype=float).reshape(-1)
    mascara = y_real > 0
    if mascara.sum() == 0:
        return {tol: np.nan for tol in tolerancias}
    error_relativo = np.abs(y_predicho[mascara] - y_real[mascara]) / y_real[mascara]
    return {tol: 100 * np.mean(error_relativo <= tol) for tol in tolerancias}

def _agregar_precio_m2_para_eda(dataframe):
    df = dataframe.copy()
    _validar_columnas_requeridas(
        df,
        ["precio", "metros_cubiertos", "Área"],
        "_agregar_precio_m2_para_eda",
    )
    superficie = df["metros_cubiertos"].where(
        df["metros_cubiertos"].notna() & (df["metros_cubiertos"] > 0),
        df["Área"],
    )
    superficie = superficie.where(superficie.notna() & (superficie > 0))
    df["precio_m2_eda"] = np.where(superficie.notna(), df["precio"] / superficie, np.nan)
    return df


def graficar_eda_precio(dataframe):
    df = dataframe.copy()
    _validar_columnas_requeridas(df, ["precio", "zona", "tipo"], "graficar_eda_precio")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9), constrained_layout=True)

    sns.histplot(
        data=df,
        x="precio",
        hue="zona",
        bins=35,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.30,
        palette=COLORES_ZONA,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Distribucion del precio observado")
    axes[0, 0].set_xlabel("precio")
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(_moneda))

    df_plot = df.assign(log_precio=_log_seguro(df["precio"]))
    sns.histplot(
        data=df_plot,
        x="log_precio",
        hue="zona",
        bins=35,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.30,
        palette=COLORES_ZONA,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Distribucion del log(1 + precio)")
    axes[0, 1].set_xlabel("log(1 + precio)")

    sns.boxplot(
        data=df_plot,
        x="zona",
        y="log_precio",
        hue="zona",
        palette=COLORES_ZONA,
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Precio por zona en escala log")
    axes[1, 0].set_xlabel("zona")
    axes[1, 0].set_ylabel("log(1 + precio)")

    sns.boxplot(
        data=df_plot,
        x="tipo",
        y="log_precio",
        hue="zona",
        palette=COLORES_ZONA,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Precio por tipo y zona")
    axes[1, 1].set_xlabel("tipo")
    axes[1, 1].set_ylabel("log(1 + precio)")
    axes[1, 1].legend(title="zona", fontsize=8)
    return fig


def graficar_dispersion_precio_area_por_zona(dataframe):
    df = dataframe.copy()
    _validar_columnas_requeridas(df, ["precio", "zona", "Área"], "graficar_dispersion_precio_area_por_zona")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), constrained_layout=True, sharey=True)

    for ax, zona in zip(axes, ["buenos_aires", "new_york"]):
        tabla = df[df["zona"] == zona].copy()
        if tabla.empty:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.axis("off")
            continue
        sns.scatterplot(
            data=tabla,
            x="Área",
            y="precio",
            color=COLORES_ZONA.get(zona, "#7FB3D5"),
            alpha=0.55,
            s=28,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title(f"Precio vs Area - {zona.replace('_', ' ').title()}")
        ax.set_xlabel("Area")
        ax.set_ylabel("precio (escala log)")

    return fig


def graficar_precio_m2_por_zona_y_tipo(dataframe):
    df = _agregar_precio_m2_para_eda(dataframe)
    _validar_columnas_requeridas(df, ["zona", "tipo", "precio_m2_eda"], "graficar_precio_m2_por_zona_y_tipo")
    df = df.loc[df["precio_m2_eda"].notna() & (df["precio_m2_eda"] > 0)].copy()
    df["log_precio_m2"] = _log_seguro(df["precio_m2_eda"])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), constrained_layout=True)

    sns.boxplot(
        data=df,
        x="zona",
        y="log_precio_m2",
        hue="zona",
        palette=COLORES_ZONA,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Precio por m2 por zona")
    axes[0].set_xlabel("zona")
    axes[0].set_ylabel("log(1 + precio por m2)")

    sns.boxplot(
        data=df,
        x="tipo",
        y="log_precio_m2",
        hue="zona",
        palette=COLORES_ZONA,
        ax=axes[1],
    )
    axes[1].set_title("Precio por m2 por tipo y zona")
    axes[1].set_xlabel("tipo")
    axes[1].set_ylabel("log(1 + precio por m2)")
    axes[1].legend(title="zona", fontsize=8)
    return fig


def graficar_correlaciones_por_zona(dataframe):
    df = dataframe.copy()
    _validar_columnas_requeridas(df, ["zona"], "graficar_correlaciones_por_zona")
    columnas = [
        "precio", "Área", "metros_cubiertos", "ambientes", "pisos", "edad",
        "lat", "lon", "superficie_descubierta", "proporcion_cubierta"
    ]
    columnas = [c for c in columnas if c in df.columns]
    subconjuntos = {
        "Correlacion global": df,
        "Correlacion Buenos Aires": df[df["zona"] == "buenos_aires"],
        "Correlacion New York": df[df["zona"] == "new_york"],
    }

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    grilla = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    axes = [
        fig.add_subplot(grilla[0, :]),
        fig.add_subplot(grilla[1, 0]),
        fig.add_subplot(grilla[1, 1]),
    ]
    for ax, (titulo, tabla) in zip(axes, subconjuntos.items()):
        if tabla.empty:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
            ax.set_title(titulo)
            ax.axis("off")
            continue
        corr = tabla[columnas].corr()
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            ax=ax,
            cbar=False,
            annot_kws={"size": 8},
        )
        ax.set_title(titulo)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    return fig

def graficar_curva_perdida(historial_perdida, titulo="Evolucion de la perdida"):
    historial = np.asarray(historial_perdida, dtype=float).reshape(-1)
    if historial.size == 0:
        raise ValueError("historial_perdida no puede estar vacio.")

    iteraciones = np.arange(1, historial.size + 1)
    mejor_hasta_ahora = np.minimum.accumulate(historial)
    mejora_acumulada_pct = 100 * (historial[0] - historial) / max(abs(historial[0]), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), constrained_layout=True)

    axes[0].plot(iteraciones, historial, linewidth=2.4, color="#5B8DEF", label="loss")
    axes[0].plot(
        iteraciones,
        mejor_hasta_ahora,
        linewidth=1.6,
        linestyle="--",
        color="#E57373",
        label="mejor loss acumulado",
    )
    axes[0].scatter(
        [iteraciones[0], iteraciones[-1]],
        [historial[0], historial[-1]],
        color=["#1F618D", "#AF601A"],
        s=55,
        zorder=5,
    )
    axes[0].set_yscale("log")
    axes[0].set_title(titulo)
    axes[0].set_xlabel("iteracion")
    axes[0].set_ylabel("loss (escala log)")
    axes[0].legend(loc="best", frameon=True)

    axes[1].plot(iteraciones, mejora_acumulada_pct, linewidth=2.4, color="#58B19F")
    axes[1].fill_between(iteraciones, mejora_acumulada_pct, alpha=0.18, color="#58B19F")
    axes[1].axhline(0, color="black", linewidth=1.0)
    axes[1].set_title("Mejora acumulada respecto del inicio")
    axes[1].set_xlabel("iteracion")
    axes[1].set_ylabel("mejora acumulada (%)")
    axes[1].annotate(
        f"inicio: {historial[0]:,.2f}\nfinal: {historial[-1]:,.2f}\nmejora: {mejora_acumulada_pct[-1]:.2f}%",
        xy=(iteraciones[-1], mejora_acumulada_pct[-1]),
        xytext=(-90, 14),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D5D8DC"},
        fontsize=8.5,
    )

    return fig

def graficar_m1_recta(df_validation, pred_validation, titulo="M1: ajuste lineal según Área"):
    df = df_validation.copy().reset_index(drop=True)
    pred = np.asarray(pred_validation, dtype=float).reshape(-1)
    orden = np.argsort(df["Área"].values)
    x_sorted = df["Área"].values[orden]
    y_sorted = pred[orden]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)

    for zona, color in COLORES_ZONA.items():
        mascara = df["zona"] == zona
        axes[0].scatter(df.loc[mascara, "Área"], df.loc[mascara, "precio"], s=28, alpha=0.55, c=color, label=zona)
    axes[0].plot(x_sorted, y_sorted, color="black", linewidth=2.4, label="recta ajustada")
    axes[0].set_title(titulo)
    axes[0].set_xlabel("Área")
    axes[0].set_ylabel("precio")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_moneda))
    axes[0].legend(title="zona")

    for zona, color in COLORES_ZONA.items():
        mascara = df["zona"] == zona
        axes[1].scatter(df.loc[mascara, "Área"], _log_seguro(df.loc[mascara, "precio"]), s=28, alpha=0.55, c=color, label=zona)
    axes[1].plot(x_sorted, _log_seguro(y_sorted), color="#E57373", linewidth=2.4, label="curva ajustada")
    axes[1].set_title("M1: ajuste lineal según Área (escala log)")
    axes[1].set_xlabel("Área")
    axes[1].set_ylabel("log(1 + precio)")
    axes[1].legend(title="zona")
    return fig


def graficar_real_vs_predicho_panel(y_real, y_predicho, zona=None, titulo="Real vs predicho"):
    y_real = np.asarray(y_real, dtype=float).reshape(-1)
    y_predicho = np.asarray(y_predicho, dtype=float).reshape(-1)
    bandas = _porcentajes_en_bandas(y_real, y_predicho, tolerancias=(0.10, 0.25))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)

    ax = axes[0]
    if zona is None:
        ax.scatter(y_real, y_predicho, s=22, alpha=0.5, color="#7FB3D5")
    else:
        zona = pd.Series(zona).reset_index(drop=True)
        for nombre_zona, color in COLORES_ZONA.items():
            mascara = zona == nombre_zona
            ax.scatter(y_real[mascara], y_predicho[mascara], s=22, alpha=0.55, c=color, label=nombre_zona)
        ax.legend(title="zona", loc="lower right", frameon=True)

    minimo = min(y_real.min(), y_predicho.min())
    maximo = max(y_real.max(), y_predicho.max())
    margen = 0.03 * (maximo - minimo)
    minimo -= margen
    maximo += margen
    linea = np.linspace(max(minimo, 0), maximo, 300)

    ax.plot(linea, linea, color="black", linewidth=2.2)
    ax.fill_between(linea, 0.90 * linea, 1.10 * linea, color="#D5F5E3", alpha=0.45)
    ax.fill_between(linea, 0.75 * linea, 1.25 * linea, color="#FCF3CF", alpha=0.28)
    ax.plot(linea, 0.90 * linea, color="#7DCEA0", linestyle="--", linewidth=1.2)
    ax.plot(linea, 1.10 * linea, color="#7DCEA0", linestyle="--", linewidth=1.2)
    ax.plot(linea, 0.75 * linea, color="#F7DC6F", linestyle=":", linewidth=1.2)
    ax.plot(linea, 1.25 * linea, color="#F7DC6F", linestyle=":", linewidth=1.2)
    ax.set_xlim(minimo, maximo)
    ax.set_ylim(minimo, max(maximo, 0))
    ax.set_title(titulo)
    ax.set_xlabel("precio real")
    ax.set_ylabel("precio predicho")
    ax.xaxis.set_major_formatter(FuncFormatter(_moneda))
    ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
    texto = f"Dentro de ±10%: {bandas[0.10]:.1f}%\nDentro de ±25%: {bandas[0.25]:.1f}%"
    ax.text(0.98, 0.98, texto, transform=ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#CCCCCC"), fontsize=9)

    ax = axes[1]
    xlog = _log_seguro(y_real)
    ylog = _log_seguro(y_predicho)
    if zona is None:
        ax.scatter(xlog, ylog, s=22, alpha=0.5, color="#7FB3D5")
    else:
        for nombre_zona, color in COLORES_ZONA.items():
            mascara = zona == nombre_zona
            ax.scatter(xlog[mascara], ylog[mascara], s=22, alpha=0.55, c=color, label=nombre_zona)
        ax.legend(title="zona", loc="lower right", frameon=True)

    base = np.linspace(0, max(max(y_real.max(), y_predicho.max()), 1), 300)
    ax.plot(_log_seguro(base), _log_seguro(base), color="black", linewidth=2.2)
    ax.plot(_log_seguro(base), _log_seguro(1.10 * base), color="#7DCEA0", linestyle="--", linewidth=1.2)
    ax.plot(_log_seguro(base), _log_seguro(0.90 * base), color="#7DCEA0", linestyle="--", linewidth=1.2)
    ax.plot(_log_seguro(base), _log_seguro(1.25 * base), color="#F7DC6F", linestyle=":", linewidth=1.2)
    ax.plot(_log_seguro(base), _log_seguro(0.75 * base), color="#F7DC6F", linestyle=":", linewidth=1.2)
    ax.set_title(f"{titulo} (escala log)")
    ax.set_xlabel("log(1 + precio real)")
    ax.set_ylabel("log(1 + precio predicho)")
    ax.text(0.98, 0.98, texto, transform=ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#CCCCCC"), fontsize=9)
    return fig


def graficar_residuos_panel(y_real, y_predicho, zona=None, titulo_base="Residuos"):
    y_real = np.asarray(y_real, dtype=float).reshape(-1)
    y_predicho = np.asarray(y_predicho, dtype=float).reshape(-1)
    residuos = y_real - y_predicho
    error_abs = np.abs(residuos)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    if zona is None:
        ax.scatter(y_predicho, residuos, s=22, alpha=0.5, color="#7FB3D5")
    else:
        zona = pd.Series(zona).reset_index(drop=True)
        for nombre_zona, color in COLORES_ZONA.items():
            mascara = zona == nombre_zona
            ax.scatter(y_predicho[mascara], residuos[mascara], s=22, alpha=0.55, c=color, label=nombre_zona)
        ax.legend(title="zona", loc="best")
    ax.axhline(0, color="black", linewidth=1.4)
    ax.set_title(f"{titulo_base}: residuos vs predicción")
    ax.set_xlabel("predicción")
    ax.set_ylabel("residuo")
    ax.xaxis.set_major_formatter(FuncFormatter(_moneda))
    ax.yaxis.set_major_formatter(FuncFormatter(_moneda))

    ax = axes[0, 1]
    if zona is None:
        ax.scatter(_log_seguro(y_predicho), residuos, s=22, alpha=0.5, color="#A8D5BA")
    else:
        for nombre_zona, color in COLORES_ZONA.items():
            mascara = zona == nombre_zona
            ax.scatter(_log_seguro(y_predicho[mascara]), residuos[mascara], s=22, alpha=0.55, c=color, label=nombre_zona)
        ax.legend(title="zona", loc="best")
    ax.axhline(0, color="black", linewidth=1.4)
    ax.set_title(f"{titulo_base}: residuos vs log(predicción)")
    ax.set_xlabel("log(1 + predicción)")
    ax.set_ylabel("residuo")
    ax.yaxis.set_major_formatter(FuncFormatter(_moneda))

    ax = axes[1, 0]
    sns.histplot(residuos, bins=30, kde=True, color="#C7B8EA", ax=ax)
    ax.axvline(0, color="black", linewidth=1.4)
    ax.set_title(f"{titulo_base}: distribución de residuos")
    ax.set_xlabel("residuo")
    ax.set_ylabel("frecuencia")
    ax.xaxis.set_major_formatter(FuncFormatter(_moneda))

    ax = axes[1, 1]
    if zona is None:
        sns.boxplot(y=error_abs, color="#F5B7B1", ax=ax)
        ax.set_xticks([0])
        ax.set_xticklabels(["total"])
    else:
        tabla = pd.DataFrame({"zona": zona, "error_abs": error_abs})
        sns.boxplot(data=tabla, x="zona", y="error_abs", hue="zona", palette=COLORES_ZONA, legend=False, ax=ax)
    ax.set_title(f"{titulo_base}: error absoluto por zona")
    ax.set_xlabel("Zona")
    ax.set_ylabel("error absoluto")
    ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
    return fig


def graficar_comparacion_metricas(tabla_metricas, titulo="Comparación de métricas", escenario="precio_observado"):
    tabla = tabla_metricas.copy()
    if "escenario_target" in tabla.columns:
        tabla = tabla[tabla["escenario_target"] == escenario].copy()

    modelos = tabla["modelo"].tolist()
    rmse = tabla["RMSE_validation"].values
    mae = tabla["MAE_validation"].values

    x = np.arange(len(modelos))
    ancho = 0.36
    colores = [COLORES_MODELO.get(m, "#CCCCCC") for m in modelos]

    fig, ax = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    barras1 = ax.bar(x - ancho / 2, rmse, width=ancho, color=colores, alpha=0.9, label="RMSE validation")
    barras2 = ax.bar(x + ancho / 2, mae, width=ancho, color="#F8C471", alpha=0.75, label="MAE validation")

    _anotar_barras(ax, barras1, desfase_relativo=0.015, fontsize=8)
    _anotar_barras(ax, barras2, desfase_relativo=0.015, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.set_title(titulo)
    ax.set_ylabel("valor de la métrica")
    ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
    ax.margins(y=0.18)
    ax.legend(frameon=True)
    return fig


def graficar_comparacion_ecm(tabla_metricas, titulo="Comparación de ECM", escenario="precio_observado"):
    tabla = tabla_metricas.copy()
    if "escenario_target" in tabla.columns:
        tabla = tabla[tabla["escenario_target"] == escenario].copy()

    fig, ax = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    colores = [COLORES_MODELO.get(m, "#CCCCCC") for m in tabla["modelo"]]
    barras = ax.bar(tabla["modelo"], tabla["ECM_validation"], color=colores, alpha=0.9)
    ax.set_title(titulo)
    ax.set_ylabel("ECM validation")
    _anotar_barras(ax, barras, desfase_relativo=0.02, fontsize=8)
    ax.margins(y=0.18)
    return fig


def construir_matriz_mejora_relativa(tabla_metricas, columna_metrica="RMSE_validation", escenario="precio_observado", orden_modelos=None):
    tabla = tabla_metricas.copy()
    if "escenario_target" in tabla.columns:
        tabla = tabla[tabla["escenario_target"] == escenario].copy()
    tabla = tabla.drop_duplicates(subset=["modelo"]).set_index("modelo")

    if orden_modelos is None:
        modelos = list(tabla.index)
    else:
        modelos = [m for m in orden_modelos if m in tabla.index]
        tabla = tabla.loc[modelos]

    matriz = pd.DataFrame(index=modelos, columns=modelos, dtype=float)
    for modelo_fila in modelos:
        for modelo_columna in modelos:
            metrica_fila = float(tabla.loc[modelo_fila, columna_metrica])
            metrica_columna = float(tabla.loc[modelo_columna, columna_metrica])
            matriz.loc[modelo_fila, modelo_columna] = 100 * (metrica_columna - metrica_fila) / metrica_columna
    return matriz


def graficar_matriz_mejora_relativa(matriz_mejora, titulo="Matriz de mejora relativa (%)"):
    matriz = matriz_mejora.astype(float)
    fig, ax = plt.subplots(figsize=(8.6, 6.8), constrained_layout=True)
    sns.heatmap(matriz, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.5, linecolor="white", cbar_kws={"label": "mejora relativa (%)"}, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Modelo de referencia (columna)")
    ax.set_ylabel("Modelo comparado (fila)")
    return fig


def graficar_ranking_features_candidatas(tabla_ranking, top_n=20, titulo="Top de features candidatas"):
    tabla = tabla_ranking.copy().head(top_n)
    _validar_columnas_requeridas(
        tabla,
        ["feature", "corr_con_precio", "abs_corr_con_precio"],
        "graficar_ranking_features_candidatas",
    )
    tabla = tabla.sort_values("abs_corr_con_precio", ascending=True)
    colores = ["#F4A7B9" if valor < 0 else "#A8D5BA" for valor in tabla["corr_con_precio"]]

    fig_altura = max(5.5, 0.36 * len(tabla) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_altura), constrained_layout=True)
    barras = ax.barh(tabla["feature"], tabla["corr_con_precio"], color=colores)
    ax.axvline(0, color="black", linewidth=1.1)
    ax.set_title(titulo)
    ax.set_xlabel("correlacion con precio")
    ax.set_ylabel("feature")
    for barra, valor in zip(barras, tabla["corr_con_precio"]):
        ax.text(
            valor + (0.012 if valor >= 0 else -0.012),
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.3f}",
            va="center",
            ha="left" if valor >= 0 else "right",
            fontsize=8,
        )
    return fig


def graficar_regularization_paths(tabla_coeficientes_lambdas, titulo, top_n=None):
    tabla = tabla_coeficientes_lambdas.copy()
    _validar_columnas_requeridas(
        tabla,
        ["feature", "coeficiente", "lambda"],
        "graficar_regularization_paths",
    )
    tabla = tabla[tabla["feature"] != "intercepto"].copy()
    if top_n is not None:
        ranking = (
            tabla.assign(abs_coef=np.abs(tabla["coeficiente"]))
            .groupby("feature")["abs_coef"]
            .max()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        tabla = tabla[tabla["feature"].isin(ranking)]

    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    features = sorted(tabla["feature"].unique().tolist())
    cmap = plt.get_cmap("tab20")
    colores = {feature: cmap(i % 20) for i, feature in enumerate(features)}

    for feature, grp in tabla.groupby("feature"):
        grp = grp.sort_values("lambda")
        ax.semilogx(
            grp["lambda"],
            grp["coeficiente"],
            linewidth=1.8,
            alpha=0.9,
            label=feature,
            color=colores[feature],
        )

    ax.axhline(0, color="black", linewidth=1.1)
    ax.set_title(titulo)
    ax.set_xlabel("lambda (escala log)")
    ax.set_ylabel("coeficiente")
    cantidad_features = tabla["feature"].nunique()
    if cantidad_features > 0:
        ncol = 1 if cantidad_features <= 10 else 2
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            frameon=True,
            title="feature",
            ncol=ncol,
        )
    return fig


def _graficar_curva_cv_en_eje(ax, tabla_cv, titulo, color, marcador):
    tabla = tabla_cv.copy().sort_values("lambda")
    mejor_fila = tabla.loc[tabla["ECM_promedio_cv"].idxmin()]

    ax.semilogx(
        tabla["lambda"],
        tabla["ECM_promedio_cv"],
        marker=marcador,
        linewidth=2.2,
        color=color,
        label="ECM promedio CV",
    )

    ax.axvline(mejor_fila["lambda"], color=color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.scatter(
        [mejor_fila["lambda"]],
        [mejor_fila["ECM_promedio_cv"]],
        s=80,
        color="#C0392B",
        zorder=5,
        label=f"mejor lambda = {mejor_fila['lambda']:.4g}",
    )

    if "coeficientes_casi_cero_promedio" in tabla.columns:
        tabla_con_ceros = tabla[tabla["coeficientes_casi_cero_promedio"] > 0]
        if not tabla_con_ceros.empty:
            fila_con_ceros = tabla_con_ceros.iloc[0]
            ax.axvline(
                fila_con_ceros["lambda"],
                color="#6C3483",
                linestyle=":",
                linewidth=1.1,
                alpha=0.85,
            )
            ax.scatter(
                [fila_con_ceros["lambda"]],
                [fila_con_ceros["ECM_promedio_cv"]],
                s=60,
                color="#6C3483",
                zorder=5,
                label=f"primer lambda con ceros = {fila_con_ceros['lambda']:.4g}",
            )

    ax.set_title(titulo)
    ax.set_xlabel("lambda (escala log)")
    ax.set_ylabel("ECM promedio CV")
    ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
    ax.legend(loc="best", frameon=True)
    return mejor_fila


def graficar_panel_cv_regularizacion(tabla_ridge, tabla_lasso):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), constrained_layout=True)

    mejor_ridge = _graficar_curva_cv_en_eje(
        axes[0],
        tabla_ridge,
        titulo="Ridge: ECM promedio CV vs lambda",
        color=COLORES_MODELO["Ridge"],
        marcador="o",
    )
    mejor_lasso = _graficar_curva_cv_en_eje(
        axes[1],
        tabla_lasso,
        titulo="Lasso: ECM promedio CV vs lambda",
        color=COLORES_MODELO["Lasso"],
        marcador="s",
    )

    tr = tabla_ridge.copy().sort_values("lambda")
    tl = tabla_lasso.copy().sort_values("lambda")

    axes[2].semilogx(
        tr["lambda"],
        tr["ECM_promedio_cv"],
        marker="o",
        color=COLORES_MODELO["Ridge"],
        linewidth=2.2,
        label="Ridge",
    )
    axes[2].semilogx(
        tl["lambda"],
        tl["ECM_promedio_cv"],
        marker="s",
        color=COLORES_MODELO["Lasso"],
        linewidth=2.2,
        label="Lasso",
    )

    axes[2].scatter(
        [mejor_ridge["lambda"], mejor_lasso["lambda"]],
        [mejor_ridge["ECM_promedio_cv"], mejor_lasso["ECM_promedio_cv"]],
        color=["#1F618D", "#AF601A"],
        s=90,
        zorder=5,
    )
    tabla_lasso_con_ceros = tl[tl["coeficientes_casi_cero_promedio"] > 0] if "coeficientes_casi_cero_promedio" in tl.columns else pd.DataFrame()
    if not tabla_lasso_con_ceros.empty:
        primer_lambda_con_ceros = tabla_lasso_con_ceros.iloc[0]
        axes[2].axvline(
            primer_lambda_con_ceros["lambda"],
            color="#6C3483",
            linestyle=":",
            linewidth=1.1,
            alpha=0.8,
        )
    axes[2].set_title("Comparacion conjunta de Ridge y Lasso")
    axes[2].set_xlabel("lambda (escala log)")
    axes[2].set_ylabel("ECM promedio CV")
    axes[2].yaxis.set_major_formatter(FuncFormatter(_moneda))
    axes[2].legend(loc="best", frameon=True)

    return fig


def graficar_metricas_regularizacion_vs_lambda(tabla_metricas, metricas, titulo):
    tabla = tabla_metricas.copy()
    _validar_columnas_requeridas(
        tabla,
        ["tipo_regularizacion", "lambda", *metricas],
        "graficar_metricas_regularizacion_vs_lambda",
    )

    ncols = 2
    nrows = int(np.ceil(len(metricas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)

    configuracion_series = [
        ("ridge", "Ridge", COLORES_MODELO["Ridge"], "o"),
        ("lasso", "Lasso", COLORES_MODELO["Lasso"], "s"),
    ]
    etiquetas_metricas = {
        "RMSE_train": "RMSE train",
        "RMSE_validation": "RMSE validation",
        "MAE_validation": "MAE validation",
        "R2_validation": "R2 validation",
        "ECM_validation": "ECM validation",
        "cantidad_coeficientes_casi_cero": "Coeficientes casi cero",
        "gap_RMSE_validation_menos_train": "Gap RMSE (validation - train)",
    }
    metricas_monetarias = {
        "RMSE_train",
        "RMSE_validation",
        "MAE_validation",
        "ECM_validation",
        "gap_RMSE_validation_menos_train",
    }

    for ax, metrica in zip(axes, metricas):
        for tipo, etiqueta, color, marcador in configuracion_series:
            tabla_tipo = tabla[tabla["tipo_regularizacion"] == tipo].sort_values("lambda")
            if tabla_tipo.empty:
                continue

            ax.semilogx(
                tabla_tipo["lambda"],
                tabla_tipo[metrica],
                marker=marcador,
                linewidth=2.1,
                color=color,
                label=etiqueta,
            )

            if metrica == "R2_validation":
                idx_optimo = tabla_tipo[metrica].idxmax()
            elif metrica == "cantidad_coeficientes_casi_cero":
                idx_optimo = tabla_tipo["lambda"].idxmax()
            else:
                idx_optimo = tabla_tipo[metrica].idxmin()

            fila_optima = tabla_tipo.loc[idx_optimo]
            ax.scatter(
                [fila_optima["lambda"]],
                [fila_optima[metrica]],
                s=70,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )

        etiqueta = etiquetas_metricas.get(metrica, metrica.replace("_", " "))
        ax.set_title(etiqueta)
        ax.set_xlabel("lambda (escala log)")
        ax.set_ylabel(etiqueta)
        if metrica in metricas_monetarias:
            ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
        ax.legend(loc="best", frameon=True)

    for ax in axes[len(metricas):]:
        ax.axis("off")

    fig.suptitle(titulo, fontsize=15, weight="semibold")
    return fig


def graficar_complejidad_features(tabla_curva, titulo="Complejidad vs generalizacion"):
    tabla = tabla_curva.copy().sort_values("cantidad_features_numericas")
    _validar_columnas_requeridas(
        tabla,
        [
            "cantidad_features_numericas",
            "RMSE_train",
            "RMSE_validation",
            "gap_RMSE_validation_menos_train",
        ],
        "graficar_complejidad_features",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.2), constrained_layout=True)

    axes[0].plot(
        tabla["cantidad_features_numericas"],
        tabla["RMSE_train"],
        marker="o",
        linewidth=2.2,
        color="#5B8DEF",
        label="train",
    )
    axes[0].plot(
        tabla["cantidad_features_numericas"],
        tabla["RMSE_validation"],
        marker="s",
        linewidth=2.2,
        color="#E57373",
        label="validation",
    )
    mejor_fila = tabla.loc[tabla["RMSE_validation"].idxmin()]
    axes[0].scatter(
        [mejor_fila["cantidad_features_numericas"]],
        [mejor_fila["RMSE_validation"]],
        s=90,
        color="#AF601A",
        zorder=5,
        label="mejor validation",
    )
    axes[0].set_title("RMSE vs cantidad de features")
    axes[0].set_xlabel("cantidad de features numericas")
    axes[0].set_ylabel("RMSE")
    axes[0].yaxis.set_major_formatter(FuncFormatter(_moneda))
    axes[0].legend(loc="best", frameon=True)

    axes[1].plot(
        tabla["cantidad_features_numericas"],
        tabla["gap_RMSE_validation_menos_train"],
        marker="o",
        linewidth=2.2,
        color="#58B19F",
    )
    axes[1].axhline(0, color="black", linewidth=1.0)
    axes[1].set_title("Gap de generalizacion")
    axes[1].set_xlabel("cantidad de features numericas")
    axes[1].set_ylabel("RMSE validation - RMSE train")
    axes[1].yaxis.set_major_formatter(FuncFormatter(_moneda))

    fig.suptitle(titulo, fontsize=15, weight="semibold")
    return fig


def graficar_importancia_coeficientes(tabla_coeficientes, titulo, top_n=15):
    tabla = tabla_coeficientes.copy()
    tabla = tabla[tabla["feature"] != "intercepto"].copy()
    tabla = tabla.sort_values("abs_coeficiente", ascending=False).head(top_n)
    tabla = tabla.sort_values("coeficiente", ascending=True)
    colores = ["#F4A7B9" if c < 0 else "#A8D5BA" for c in tabla["coeficiente"]]

    fig_altura = max(5, 0.35 * len(tabla) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_altura), constrained_layout=True)
    barras = ax.barh(tabla["feature"], tabla["coeficiente"], color=colores)
    ax.axvline(0, color="black", linewidth=1.1)
    ax.set_title(titulo)
    ax.set_xlabel("coeficiente")
    ax.set_ylabel("feature")
    amplitud = np.nanmax(np.abs(tabla["coeficiente"].values)) if len(tabla) else 0.0
    desplazamiento = amplitud * 0.03 if amplitud > 0 else 0.0
    for barra, valor in zip(barras, tabla["coeficiente"]):
        posicion_x = valor + desplazamiento if valor >= 0 else valor - desplazamiento
        alineacion = "left" if valor >= 0 else "right"
        ax.text(
            posicion_x,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:,.1f}".replace(",", "."),
            va="center",
            ha=alineacion,
            fontsize=8,
        )
    return fig


def graficar_metricas_validacion(tabla_modelos, titulo="Comparación de modelos en validation"):
    tabla = tabla_modelos.copy()
    metricas = ["ECM_validation", "MAE_validation", "RMSE_validation"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, metrica in zip(axes, metricas):
        sns.barplot(data=tabla, x="modelo", y=metrica, hue="modelo", palette="pastel", legend=False, ax=ax)
        ax.set_title(metrica.replace("_", " "))
        ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
        ax.margins(y=0.18)
    fig.suptitle(titulo, fontsize=15, weight="semibold")
    return fig


def graficar_learning_curves_modelos(tabla_learning_curves, metrica="RMSE", titulo="Learning curves por modelo"):
    tabla = tabla_learning_curves.copy()
    modelos = tabla["modelo"].drop_duplicates().tolist()
    modelos = sorted(
        modelos,
        key=lambda nombre: int(nombre[1:]) if nombre.startswith("M") and nombre[1:].isdigit() else 999,
    )

    ncols = 3
    nrows = int(np.ceil(len(modelos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, modelo in zip(axes, modelos):
        tabla_modelo = tabla[tabla["modelo"] == modelo].sort_values("n_train")
        ax.plot(
            tabla_modelo["n_train"],
            tabla_modelo[f"{metrica}_train"],
            marker="o",
            linewidth=2.0,
            color="#5B8DEF",
            label="train",
        )
        ax.plot(
            tabla_modelo["n_train"],
            tabla_modelo[f"{metrica}_validation"],
            marker="s",
            linewidth=2.0,
            color="#E57373",
            label="validation",
        )
        ax.set_title(modelo)
        ax.set_xlabel("cantidad de ejemplos de train")
        ax.set_ylabel(metrica)
        if metrica in {"RMSE", "MAE"}:
            ax.yaxis.set_major_formatter(FuncFormatter(_moneda))
        ax.legend(loc="best", frameon=True)

    for ax in axes[len(modelos):]:
        ax.axis("off")

    fig.suptitle(titulo, fontsize=15, weight="semibold")
    return fig
