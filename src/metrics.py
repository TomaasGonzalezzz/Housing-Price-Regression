"""Metricas de regresion."""

import numpy as np


def mse(y_true, y_pred):
    """Error cuadratico medio."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    """Error absoluto medio."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Raiz del error cuadratico medio."""
    return np.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    """Coeficiente R2."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


def resumir_metricas_regresion(y_true, y_pred):
    """Devuelve las metricas principales en un unico diccionario."""
    return {
        "ECM": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
