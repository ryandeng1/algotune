import numpy as np
from numba import njit

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        X = np.array(problem["X"])
        n_components = problem["n_components"]
        m, n = X.shape

        np.random.seed(0)
        W = np.random.rand(m, n_components)
        H = np.random.rand(n_components, n)

        @njit
        def update():
            for _ in range(100):
                for i in range(m):
                    for j in range(n_components):
                        numerator = 0.0
                        denominator = 0.0
                        for k in range(n):
                            numerator += X[i, k] * H[k, j]
                            denominator += X[i, k] * H[k, j] * W[i, j]
                        W[i, j] = W[i, j] * numerator / denominator
                
                for i in range(n_components):
                    for j in range(n):
                        numerator = 0.0
                        denominator = 0.0
                        for k in range(m):
                            numerator += W[k, i] * X[k, j]
                            denominator += W[k, i] * X[k, j] * H[i, j]
                        H[i, j] = H[i, j] * numerator / denominator

        update()

        return {"W": W.tolist(), "H": H.tolist()}
