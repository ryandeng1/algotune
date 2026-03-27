from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[float]]]]:
        """
        Compute the reduced QR factorization of the input matrix.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing the key ``"matrix"`` with a 2‑D numpy array.

        Returns
        -------
        dict[str, dict[str, list[list[float]]]]
            Nested dictionary containing the QR factorization:
            ``{"QR": {"Q": Q, "R": R}}``, where ``Q`` and ``R`` are
            2‑D lists of floats.
        """
        # Grab the input matrix
        A = problem["matrix"]

        # Use numpy's efficient LAPACK implementation to obtain the reduced QR
        Q, R = np.linalg.qr(A, mode="reduced")

        # Convert the numpy arrays to Python lists for the expected output type
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}