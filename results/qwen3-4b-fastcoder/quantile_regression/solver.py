from typing import Any
import numpy as np
from sklearn.linear_model import QuantileRegressor


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)

        model = QuantileRegressor(
            quantile=problem["quantile"],
            alpha=0.0,
            fit_intercept=problem["fit_intercept"],
            solver="highs",
        )
        model.fit(X, y)

        coef = model.coef_.tolist()
        intercept = [model.intercept_]
        predictions = (np.dot(X, model.coef_) + model.intercept_).tolist()

        return {"coef": coef, "intercept": intercept, "predictions": predictions}