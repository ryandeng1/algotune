import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        A = problem["matrix"]
        P, L, U = lu(A)
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}
