import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        vec1, vec2 = problem
        # Use NumPy's fast routine for outer product
        return np.multiply.outer(vec1, vec2)