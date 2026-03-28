from typing import Any, List
import numpy as np
from sklearn.linear_model import Lasso

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """Fit a Lasso regression (alpha = 0.1) to X and y.
        If parameters are invalid, return zeros of appropriate dimension.
        """
        X = problem.get("X")
        y = problem.get("y")
        if X is None or y is None:
            # Missing data, return zeros
            try:
                d = problem["X"].shape[1]
            except Exception:
                d = 0
            return [0.0] * d

        try:
            model = Lasso(alpha=0.1, fit_intercept=False, max_iter=10000)
            model.fit(X, y)
            return model.coef_.tolist()
        except Exception:
            # On error (e.g., shape mismatch), return zeros
            try:
                d = X.shape[1]
            except Exception:
                d = 0
            return [0.0] * d