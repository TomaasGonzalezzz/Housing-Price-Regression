"""Regresion lineal implementada con NumPy."""

import numpy as np
import pandas as pd


class RegresionLineal:
    """Modelo lineal con OLS, Ridge y Lasso."""

    def __init__(
        self,
        X,
        y,
        feature_names=None,
        fit_intercept=True,
        l2=0.0,
        l1=0.0,
    ):
        self.X = self._to_2d_array(X)
        self.y = self._to_1d_array(y)

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X e y deben tener la misma cantidad de filas.")

        self.fit_intercept = bool(fit_intercept)
        self.l2 = float(l2)
        self.l1 = float(l1)

        if feature_names is None:
            self.feature_names = [f"x_{i}" for i in range(self.X.shape[1])]
        else:
            self.feature_names = list(feature_names)

        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("La cantidad de feature_names debe coincidir con las columnas de X.")

        self.coef = None
        self.loss_history = []

    def _to_2d_array(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _to_1d_array(self, y):
        return np.asarray(y, dtype=float).reshape(-1)

    def _add_intercept_column(self, X):
        X = self._to_2d_array(X)
        if not self.fit_intercept:
            return X

        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def _coeficientes_sin_intercepto(self, pesos):
        if self.fit_intercept:
            return pesos[1:]
        return pesos

    
    def fit_pseudoinverse(self):
        """Ajusta el modelo por solucion cerrada."""
        if self.l1 != 0:
            raise ValueError("No existe solucion analitica cerrada cuando l1 es distinto de cero.")

        X_design = self._add_intercept_column(self.X)

        if self.l2 == 0:
            self.coef = np.linalg.pinv(X_design) @ self.y
            return self.coef

        n_features_with_bias = X_design.shape[1]
        identidad = np.eye(n_features_with_bias)
        if self.fit_intercept:
            identidad[0, 0] = 0.0

        xtx = X_design.T @ X_design
        xty = X_design.T @ self.y
        self.coef = np.linalg.pinv(xtx + self.l2 * identidad) @ xty
        return self.coef

    def _compute_gradient(self, X_design, y_true, weights):
        n_samples = X_design.shape[0]

        predictions = X_design @ weights
        errors = predictions - y_true

        gradient = (2.0 / n_samples) * (X_design.T @ errors)

        if self.l2 != 0:
            reg_l2 = np.zeros_like(weights)
            reg_l2_slice = self._coeficientes_sin_intercepto(weights)
            if self.fit_intercept:
                reg_l2[1:] = reg_l2_slice
            else:
                reg_l2[:] = reg_l2_slice
            gradient += 2.0 * self.l2 * reg_l2

        return gradient

    def _compute_loss(self, X_design, y_true, weights):
        predictions = X_design @ weights
        mse_value = np.mean((y_true - predictions) ** 2)

        weights_no_bias = self._coeficientes_sin_intercepto(weights)
        l2_penalty = self.l2 * np.sum(weights_no_bias ** 2) if self.l2 != 0 else 0.0
        l1_penalty = self.l1 * np.sum(np.abs(weights_no_bias)) if self.l1 != 0 else 0.0

        return mse_value + l2_penalty + l1_penalty

    def _apply_l1_step(self, weights, learning_rate):
        if self.l1 == 0:
            return weights

        updated_weights = weights.copy()
        threshold = learning_rate * self.l1

        if self.fit_intercept:
            penalized_slice = slice(1, None)
        else:
            penalized_slice = slice(None)

        penalized_weights = updated_weights[penalized_slice]
        updated_weights[penalized_slice] = np.sign(penalized_weights) * np.maximum(
            np.abs(penalized_weights) - threshold,
            0.0,
        )
        return updated_weights

    def fit_gradient_descent(
        self,
        learning_rate=0.01,
        n_iterations=5000,
        tolerance=1e-8,
        verbose=False,
        initial_weights=None,
    ):
        """Ajusta el modelo con descenso por gradiente."""
        X_design = self._add_intercept_column(self.X)
        n_features_with_bias = X_design.shape[1]

        if initial_weights is None:
            weights = np.zeros(n_features_with_bias)
        else:
            weights = np.asarray(initial_weights, dtype=float).reshape(-1)
            if weights.shape[0] != n_features_with_bias:
                raise ValueError("initial_weights tiene dimension incorrecta.")

        self.loss_history = []
        previous_loss = None

        for iteration in range(n_iterations):
            gradient = self._compute_gradient(X_design, self.y, weights)
            weights = weights - learning_rate * gradient
            weights = self._apply_l1_step(weights, learning_rate)

            current_loss = self._compute_loss(X_design, self.y, weights)
            self.loss_history.append(current_loss)

            if verbose and iteration % 500 == 0:
                print(f"Iteracion {iteration}: loss = {current_loss:.6f}")

            if previous_loss is not None and abs(previous_loss - current_loss) < tolerance:
                if verbose:
                    print(f"Convergencia alcanzada en iteracion {iteration}.")
                break

            previous_loss = current_loss

        self.coef = weights
        return self.coef

    def predict(self, X):
        if self.coef is None:
            raise ValueError("El modelo todavia no fue entrenado.")

        X_design = self._add_intercept_column(X)
        return X_design @ self.coef

    def mse(self, X, y):
        y_true = self._to_1d_array(y)
        y_pred = self.predict(X)
        return np.mean((y_true - y_pred) ** 2)

    def get_coefficient_dict(self):
        if self.coef is None:
            raise ValueError("El modelo todavia no fue entrenado.")

        coef_dict = {}
        if self.fit_intercept:
            coef_dict["intercepto"] = self.coef[0]
            coef_values = self.coef[1:]
        else:
            coef_values = self.coef

        for name, value in zip(self.feature_names, coef_values):
            coef_dict[name] = value

        return coef_dict

    def tabla_coeficientes(self):
        """Devuelve los coeficientes ordenados por valor absoluto."""
        coef_dict = self.get_coefficient_dict()
        tabla = pd.DataFrame(
            {
                "feature": list(coef_dict.keys()),
                "coeficiente": list(coef_dict.values()),
            }
        )
        tabla["abs_coeficiente"] = np.abs(tabla["coeficiente"])
        return tabla.sort_values("abs_coeficiente", ascending=False).reset_index(drop=True)

    def print_coefficients(self):
        if self.coef is None:
            raise ValueError("El modelo todavia no fue entrenado.")

        if self.fit_intercept:
            print(f"intercepto: {self.coef[0]:.6f}")
            coef_values = self.coef[1:]
        else:
            coef_values = self.coef

        for name, value in zip(self.feature_names, coef_values):
            print(f"{name}: {value:.6f}")

    def contar_coeficientes_casi_cero(self, tolerancia=1e-6):
        """Cuenta cuantos coeficientes quedan practicamente anulados."""
        if self.coef is None:
            raise ValueError("El modelo todavia no fue entrenado.")

        pesos = self._coeficientes_sin_intercepto(self.coef)
        return int(np.sum(np.abs(pesos) <= tolerancia))
