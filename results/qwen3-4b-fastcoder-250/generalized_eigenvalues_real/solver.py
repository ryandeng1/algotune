import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> list[float]:
        A, B = problem
        n = A.shape[0]
        
        L = np.linalg.cholesky(B)
        Linv = np.linalg.solve(L, np.eye(n))
        Atilde = Linv.T @ A @ Linv
        eigenvalues = np.linalg.eigh(Atilde)[0]
        return eigenvalues[::-1].tolist()
