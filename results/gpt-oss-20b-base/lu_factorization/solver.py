import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Perform an LU factorization of A (A = P L U) and return the components.
        """
        A = problem["matrix"]
        P, L, U = lu(A)
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}