import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> list[float]:
        A, B = problem
        n = A.shape[0]
        L = np.linalg.cholesky(B)
        L_inv = np.linalg.solve(L, np.eye(n))
        Atilde = L_inv @ A @ L_inv.T
        eigenvalues = np.linalg.eigh(Atilde)[0]
        return sorted(eigenvalues, reverse=True)
