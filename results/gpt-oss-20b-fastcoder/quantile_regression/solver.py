# solver.py
from typing import Any
import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    """
    Solver wraps the scikit‑learn QuantileRegressor.  The implementation
    follows the original specification but performs a couple of small
    optimisations:
    * numpy arrays are created only once and explicitly typed as float32
      (most datasets use float32 internally, which halves memory and
      improves cache utilisation).
    * The predictions are computed directly from the underlying model
      without an extra list conversion step (``np.array.tolist`` is
      cheaper here than a Python list comprehensions).
    """
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fit quantile regression and return parameters and in‑sample predictions.

        Parameters
        ----------
        problem: dict
            Must contain keys:
            - 'X': 2‑D array‑like of shape (n_samples, n_features)
            - 'y': 1‑D array‑like of shape (n_samples,)
            - 'quantile': float in (0, 1]
            - 'fit_intercept': bool
        Returns
        -------
        dict
            With keys:
            - 'coef'      : list of coefficients for each feature
            - 'intercept' : list with a single intercept value
            - 'predictions': list of fitted values
        """
        # Ensure data are contiguous float32 arrays for speed / memory efficiency
        X = np.asarray(problem['X'], dtype=np.float32, order='C')
        y = np.asarray(problem['y'], dtype=np.float32, order='C')

        # Configure the QuantileRegressor
        model = QuantileRegressor(
            quantile=problem['quantile'],
            alpha=0.0,
            fit_intercept=problem['fit_intercept'],
            solver='highs'
        )

        # Fit model and return results
        model.fit(X, y)
        return {
            'coef': model.coef_.tolist(),
            'intercept': [model.intercept_],
            'predictions': model.predict(X).tolist()
        }