from typing import Any
import numpy as np
from sklearn.linear_model import QuantileRegressor

class Solver:
    """Fast quantile regression solver based on scikit‑learn."""
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs only once
        X = np.array(problem['X'], dtype=float, copy=False)
        y = np.array(problem['y'], dtype=float, copy=False)

        # Instantiate a lightning‑fast solver (Highs) and fit the model
        model = QuantileRegressor(
            quantile=problem['quantile'],
            alpha=0.0,
            fit_intercept=problem['fit_intercept'],
            solver='highs',
        )
        model.fit(X, y)

        # Materialise results
        coef = model.coef_.tolist()
        intercept = [model.intercept_]
        predictions = model.predict(X).tolist()

        return {"coef": coef, "intercept": intercept, "predictions": predictions}