import numpy as np
from scipy.linalg import lu_factor

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem["matrix"]
        lu, piv = lu_factor(A)
        n = A.shape[0]
        P = np.eye(n)
        for i in range(n):
            P[i, piv[i]] = 1.0
        L = lu[:n, :n]
        U = lu[n:, :n]
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
