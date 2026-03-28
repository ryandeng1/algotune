import numpy as np
from scipy.linalg import eigh
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for symmetric A and positive‑definite B.
        Returns the eigenvalues sorted in descending order.
        """
        A, B = problem
        # Use scipy's optimized generalized eigenvalue routine (eigh handles symmetric-definite pairs).
        vals, _ = eigh(A, B, eigvals_only=True, lower=False)
        # eigh returns ascending order; reverse for descending
        return vals[::-1].tolist()