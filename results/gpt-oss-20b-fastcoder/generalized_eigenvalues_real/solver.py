import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        """
        Solve the generalized eigenvalue problem A x = λ B x for symmetric A and SPD B.
        Returns eigenvalues sorted in descending order.
        """
        A, B = problem
        # Use SciPy's generalized eigenvalue routine for symmetric-definite problems
        evals, _ = eigh(A, B, eigvals_only=True, subset_by_index=[0, -1])
        # eigh returns values sorted in ascending order
        return evals[::-1].tolist()