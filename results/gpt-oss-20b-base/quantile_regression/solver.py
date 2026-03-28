import numpy as np
from sklearn.linear_model import QuantileRegressor
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Prepare data
        X = np.asarray(problem['X'], dtype=float)
        y = np.asarray(problem['y'], dtype=float)

        # Initialize quantile regressor
        model = QuantileRegressor(
            quantile=problem['quantile'],
            alpha=0.0,
            fit_intercept=problem['fit_intercept'],
            solver='highs'
        )

        # Fit and predict
        model.fit(X, y)
        coef = model.coef_.tolist()
        intercept = [model.intercept_]
        predictions = model.predict(X).tolist()

        return {'coef': coef, 'intercept': intercept, 'predictions': predictions}