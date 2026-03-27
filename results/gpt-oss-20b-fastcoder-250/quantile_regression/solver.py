import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    def solve(self, problem):
        """
        Fit quantile regression with scikit‑learn and return parameters
        + in‑sample predictions.

        :param problem: dict returned by generate_problem
        :return: dict with 'coef', 'intercept', 'predictions'
        """
        # Convert input data to contiguous float arrays
        X = np.asarray(problem["X"], dtype=np.float64, order="C")
        y = np.asarray(problem["y"], dtype=np.float64, order="C")

        # Fit quantile regression
        model = QuantileRegressor(
            quantile=problem["quantile"],
            alpha=0.0,           # no ℓ₂ regularization
            fit_intercept=problem["fit_intercept"],
            solver="highs",      # fast, requires SciPy ≥1.6
        )
        model.fit(X, y)

        # Prepare the output in the expected format
        return {
            "coef": model.coef_.tolist(),
            "intercept": [model.intercept_],
            "predictions": model.predict(X).tolist(),
        }