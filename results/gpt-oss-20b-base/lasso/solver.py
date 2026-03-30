# solver.py
from typing import Any, List

import numpy as np
from sklearn.linear_model import Lasso


class Solver:
    """
    A simple solver that fits a Lasso regression model to the input data.
    Returns the learned coefficients as a list. In case of any exception
    (e.g. malformed input), returns a zero vector of appropriate length.
    """

    def solve(self, problem: dict[str, Any]) -> List[float]:
        X = problem.get("X")
        y = problem.get("y")

        # Basic validation
        if X is None or y is None:
            return []

        try:
            # Use a very modest regularisation to obtain a sparse solution.
            clf = Lasso(alpha=0.1, fit_intercept=False, max_iter=1000, tol=1e-4)
            clf.fit(X, y)
            return clf.coef_.tolist()
        except Exception:  # pragma: no cover
            # Fallback to a zero vector if the fit fails
            d = X.shape[1] if hasattr(X, "shape") else 0
            return [0.0] * d