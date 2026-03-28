from typing import Any, Tuple, List
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray]) -> List[float]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for symmetric A and
        symmetric positive‑definite B.

        The algorithm transforms the problem into a standard eigenvalue problem
        by applying the Cholesky factor of B and then solves it with
        np.linalg.eigh.  Triangular solves are used instead of full matrix
        inversion for speed and numerical stability.

        :param problem: Tuple (A, B) where A is symmetric and B is SPD.
        :return: List of eigenvalues sorted in descending order.
        """
        A, B = problem
        # Cholesky factorization: B = L @ L.T
        L = np.linalg.cholesky(B)

        # Compute L⁻¹ A L⁻T without forming the inverse explicitly
        # Step 1: solve L X = A  →  X = L⁻¹ A
        X = solve_triangular(L, A, lower=True, trans='N', overwrite_b=False)

        # Step 2: solve L.T Atilde = X.T  →  Atilde = L⁻T X
        Atilde = solve_triangular(L.T, X.T, lower=False, trans='N', overwrite_b=False).T

        # Eigenvalues of the standard problem
        eigenvalues = np.linalg.eigh(Atilde, lower=True, eigvals_only=True)

        # Return in descending order
        return sorted(eigenvalues.tolist(), reverse=True)