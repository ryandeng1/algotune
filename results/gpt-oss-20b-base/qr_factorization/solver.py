import numpy as np
from typing import Any


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the reduced QR factorization of the matrix ``A`` from ``problem``.
        The result is returned as a dictionary with keys ``"Q"`` and ``"R"`` converted
        to Python nested lists for easy JSON serialisation or further processing.
        """
        A = problem["matrix"]
        # np.linalg.qr is already highly optimised; no further manual optimisation is needed.
        Q, R = np.linalg.qr(A, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}