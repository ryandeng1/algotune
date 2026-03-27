from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the Cholesky decomposition of the square, symmetric, positive‑definite matrix
        stored under the key ``"matrix"`` in *problem*.

        The result is returned as a dictionary
        ``{"Cholesky": {"L": L}}`` where ``L`` is a list of lists of *float* values
        representing the lower‑triangular Cholesky factor.
        """
        A = problem["matrix"]
        # NumPy already provides a highly optimised routine for Cholesky.
        L = np.linalg.cholesky(A).tolist()
        return {"Cholesky": {"L": L}}