from typing import Any
import numpy as np
from scipy.linalg import lu

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the LU factorization of the matrix A in `problem`.
        Returns a dictionary with the permutation matrix P, lower triangle L,
        and upper triangle U as plain Python lists for compatibility with
        downstream consumers.
        """
        A = problem["matrix"]
        P, L, U = lu(A)
        return {"LU": {"P": P.tolist(), "L": L.tolist(), "U": U.tolist()}}