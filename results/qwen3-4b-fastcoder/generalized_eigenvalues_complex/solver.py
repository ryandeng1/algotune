from typing import Any
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[complex]:
        A, B = problem

        scale_B = np.sqrt(np.linalg.norm(B))
        B_scaled = B / scale_B
        A_scaled = A / scale_B

        eigenvalues, _ = la.eig(A_scaled, B_scaled)

        key = np.column_stack((-eigenvalues.real, -eigenvalues.imag))
        sorted_indices = np.argsort(key)
        return eigenvalues[sorted_indices].tolist()