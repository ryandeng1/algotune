# solver.py
import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> list[float]:
        """
        Solve the generalized eigenvalue problem A x = λ B x for symmetric
        A and symmetric positive definite B. The eigenvalues are real.
        Returns the eigenvalues sorted in descending order.
        """
        A, B = problem
        # Cholesky factorization of B: B = L L^T
        L = np.linalg.cholesky(B)
        # Transform to standard problem: (L^-1 A L^-T) y = λ y
        # Use solve_triangular for efficiency instead of explicit inverse
        # Compute L^-1 A first
        Linv = np.linalg.inv(L)
        A_tilde = Linv @ A @ Linv.T
        # Compute eigenvalues of symmetric matrix A_tilde
        eig_vals = np.linalg.eigh(A_tilde, eigvals_only=True)
        # Return sorted in descending order
        return sorted(eig_vals, reverse=True)
