import numpy as np
from numba import njit

@njit
def lasso_coordinate_descent(X, y, alpha, max_iter=1000):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(max_iter):
        residual = y - np.dot(X, w)
        for j in range(d):
            grad = np.dot(X[:, j], residual) / n
            threshold = alpha / n
            if grad > threshold:
                w[j] = grad - threshold
            elif grad < -threshold:
                w[j] = grad + threshold
            else:
                w[j] = 0
    return w

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        return lasso_coordinate_descent(X, y, 0.1).tolist()
