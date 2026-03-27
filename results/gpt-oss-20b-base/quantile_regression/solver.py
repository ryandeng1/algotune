from typing import Any
import numpy as np
from sklearn.linear_model import QuantileRegressor


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fit quantile regression with scikit-learn and return parameters +
        in-sample predictions.

        :param problem: dict returned by generate_problem
        :return: dict with 'coef', 'intercept', 'predictions'
        """
        # Convert data to contiguous float arrays for fast access
        X = np.asarray(problem["X"], dtype=np.float64, order="C")
        y = np.asarray(problem["y"], dtype=np.float64, order="C")

        # Initialize the quantile regressor with an efficient solver
        model = QuantileRegressor(
            quantile=problem["quantile"],
            alpha=0.0,                # no ℓ₂ regularisation
            fit_intercept=problem["fit_intercept"],
            solver="highs",           # fast interior‑point solver (SciPy ≥ 1.6)
        )

        # Fit once, reuse underlying CG and matrix operations
        model.fit(X, y)

        # Extract model parameters and predictions in the required format
        return {
            "coef": model.coef_.tolist(),
            "intercept": [model.intercept_],
            "predictions": model.predict(X).tolist(),
        }