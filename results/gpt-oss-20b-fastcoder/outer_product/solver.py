import numpy as np
from typing import Tuple

class Solver:
    """
    Solver for computing the outer product of two 1‑D numpy arrays.

    The implementation uses NumPy's broadcasting which is a thin C layer
    that performs the multiplication element‑wise without creating an
    intermediate full outer product matrix.  This results in lower
    memory usage and faster execution compared to the high‑level
    :func:`numpy.outer` for large vectors.
    """

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        problem : tuple[np.ndarray, np.ndarray]
            A tuple containing two 1‑D numpy arrays ``vec1`` and ``vec2``.

        Returns
        -------
        np.ndarray
            The outer product of the input arrays (shape ``(len(vec1), len(vec2))``).
        """
        vec1, vec2 = problem
        # Ensure the inputs are 1‑D
        vec1 = np.asarray(vec1).ravel()
        vec2 = np.asarray(vec2).ravel()

        # Broadcasting multiplication yields the outer product
        return vec1[:, None] * vec2