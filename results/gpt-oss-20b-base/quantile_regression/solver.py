import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Fit a quantile regression model and return coefficients, intercept,
        and in-sample predictions.
        """
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)

        model = QuantileRegressor(
            quantile=problem["quantile"],
            alpha=0.0,                     # no regularisation
            fit_intercept=problem["fit_intercept"],
            solver="highs",                # fast interior-point solver
        )
        model.fit(X, y)

        return {
            "coef": model.coef_.tolist(),
            "intercept": [float(model.intercept_)],
            "predictions": model.predict(X).tolist(),
        }
