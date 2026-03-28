from typing import Any
import numpy as np
from sklearn import linear_model


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        X = problem["X"]
        y = problem["y"]
        
        if not hasattr(X, 'shape') or len(X.shape) != 2:
            d = X.shape[1] if hasattr(X, 'shape') else 0
            return np.zeros(d).tolist()
        
        if not hasattr(y, 'shape') or len(y.shape) != 1:
            return np.zeros(X.shape[1]).tolist()
        
        try:
            clf = linear_model.Lasso(alpha=0.1, fit_intercept=False)
            clf.fit(X, y)
            return clf.coef_.tolist()
        except Exception:
            return np.zeros(X.shape[1]).tolist()