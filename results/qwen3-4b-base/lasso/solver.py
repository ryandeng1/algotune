import numpy as np
from sklearn.linear_model import LassoLars

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        try:
            X = np.array(problem["X"])
            y = np.array(problem["y"])
            clf = LassoLars(alpha=0.1, fit_intercept=False)
            clf.fit(X, y)
            return clf.coef_.tolist()
        except Exception as e:
            _, d = problem["X"].shape
            return np.zeros(d).tolist()
