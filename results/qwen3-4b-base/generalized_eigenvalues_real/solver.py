from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        A, B = problem
        L = np.linalg.cholesky(B)
        Linv = np.linalg.inv(L)
        Atilde = Linv @ A @ Linv.T
        eigenvalues = np.linalg.eigh(Atilde)[0]
        return eigenvalues[::-1]