from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the Cholesky factorization of a symmetric positive‑definite matrix.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing a single key ``'matrix'`` with the input matrix.

        Returns
        -------
        dict[str, dict[str, list[list[float]]]]
            Dictionary with the lower triangular factor ``L``.
        """
        # Directly compute the Cholesky factor; no extra copies or loops.
        L = np.linalg.cholesky(problem['matrix'])
        return {"Cholesky": {"L": L}}