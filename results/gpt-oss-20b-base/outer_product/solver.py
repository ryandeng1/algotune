from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        # Unpack input vectors
        a, b = problem

        # Ensure inputs are 1‑D arrays with the same dtype for optimal broadcasting
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()

        # Compute the outer product using NumPy broadcasting.
        # This is usually faster than np.outer for small to medium sized arrays.
        return a[:, None] * b