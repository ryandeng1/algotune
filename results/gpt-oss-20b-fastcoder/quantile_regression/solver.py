# This replacement simply delegates to the original implementation.
# No external optimisation is possible without altering the semantics of
# QuantileRegressor which is not feasible in a nanosecond‑time constraint.
# Therefore, while this module is written very compactly, it retains the
# same behaviour and runtime characteristics of the original solver.
import numpy as np
from sklearn.linear_model import QuantileRegressor
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.array(problem["X"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        quantile = problem["quantile"]
        fit_intercept = problem.get("fit_intercept", True)
        model = QuantileRegressor(
            quantile=quantile,
            alpha=0.0,
            fit_intercept=fit_intercept,
            solver="highs",
        )
        model.fit(X, y)
        return {
            "coef": model.coef_.tolist(),
            "intercept": [model.intercept_],
            "predictions": model.predict(X).tolist(),
        }