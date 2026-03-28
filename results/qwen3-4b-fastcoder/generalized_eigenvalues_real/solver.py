from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        A, B = problem
        L = np.linalg.cholesky(B)
        n = L.shape[0]
        Linv = np.linalg.solve_triangular(L, np.eye(n), lower=True)
        Atilde = Linv @ A @ Linv.T
        eigenvalues = np.linalg.eigh(Atilde)[0]
        return np.sort(eigenvalues)[::-1].tolist()