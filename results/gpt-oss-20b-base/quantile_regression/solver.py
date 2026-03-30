# solver.py
import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    """Fast quantile‑regression solver using scikit‑learn's high‑performance
    linear‑programming backend (Highs)."""

    def solve(self, problem: dict) -> dict:
        """
        Fit quantile regression and return parameters and predictions.

        Parameters
        ----------
        problem : dict
            Must contain:
            - 'X' : 2‑D sequence of features
            - 'y' : target sequence
            - 'quantile' : target quantile (0‑1)
            - 'fit_intercept' : bool
        Returns
        -------
        dict
            {'coef': list, 'intercept': [float], 'predictions': list}
        """
        # Ensure contiguous float arrays (fast Path)
        X = np.asarray(problem["X"], dtype=np.float64, order="C")
        y = np.asarray(problem["y"], dtype=np.float64, order="C")

        model = QuantileRegressor(
            quantile=problem["quantile"],
            alpha=0.0,                  # no regularisation
            fit_intercept=problem["fit_intercept"],
            solver="highs",             # fast LP solver
        )
        # Fit in place (keeps interim memory small)
        model.fit(X, y)

        # Convert to Python natives once, no intermediate copies
        return {
            "coef": model.coef_.tolist(),
            "intercept": [float(model.intercept_)],
            "predictions": model.predict(X).tolist(),
        }