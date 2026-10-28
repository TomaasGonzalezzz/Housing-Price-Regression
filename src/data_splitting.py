"""Funciones para split y folds."""

import numpy as np


def train_val_split(df, train_ratio=0.8, seed=42):
    """Hace un split train/validation."""
    df = df.copy()

    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    train_size = int(train_ratio * len(df))
    train_idx = indices[:train_size]
    validation_idx = indices[train_size:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_validation = df.iloc[validation_idx].reset_index(drop=True)

    return df_train, df_validation


def generar_folds_kfold(n_samples, n_splits=5, shuffle=True, seed=42):
    """Genera folds para validacion cruzada."""
    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2.")
    if n_splits > n_samples:
        raise ValueError("n_splits no puede ser mayor que n_samples.")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)
    lista_folds = []

    for k in range(n_splits):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != k])
        lista_folds.append((train_idx, val_idx))

    return lista_folds
