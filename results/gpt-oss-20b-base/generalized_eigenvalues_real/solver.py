import numpy as np
from typing import List, Tuple

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> List[float]:
        """
        Solve the generalized symmetric–definite eigenvalue problem
        A · x = λ B · x   with A symmetric and B symmetric positive definite.

        The algorithm uses a Cholesky factorization of B and transforms the
        problem to a standard eigenvalue problem without forming the inverse
        of the Cholesky factor.

        The returned eigenvalues are sorted in descending order.
        """
        A, B = problem

        # 1. Cholesky factorization of B: B = L @ L.T
        L = np.linalg.cholesky(B)

        # 2. Solve L @ Y = A for Y  --> Y = L^{-1} @ A
        #    Using the fact that A is symmetric we can evaluate Y = L^{-T} @ A.T
        #    but a simple solve for each column is sufficient.
        Y = np.linalg.solve(L, A)

        # 3. Transform to a standard eigenvalue problem:
        #    A_tilde = Y @ Y.T  (since A_tilde = L^{-T} @ A @ L^{-1})
        A_tilde = Y @ Y.T

        # 4. Compute eigenvalues of the symmetric matrix A_tilde
        eigenvalues, _ = np.linalg.eigh(A_tilde)

        # 5. Return eigenvalues sorted in descending order
        return list(reversed(eigenvalues))