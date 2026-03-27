from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return the outer product of two 1‑D arrays."""
        vec1, vec2 = problem
        # NumPy's outer product is already highly optimised (C‑level).
        # Using np.multiply.outer provides a cleaner literal translation.
        return np.multiply.outer(vec1, vec2)