import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return the outer product of two 1‑D arrays using vectorised broadcasting."""
        vec1, vec2 = problem
        # Use broadcasting for a small performance gain over np.outer
        return vec1[:, None] * vec2[None, :]