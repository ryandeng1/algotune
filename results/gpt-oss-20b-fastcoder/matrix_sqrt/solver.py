import numpy as np
from scipy.linalg import sqrtm

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        A = np.asarray(problem["matrix"])
        try:
            X = sqrtm(A, disp=False)
        except Exception:
            return {"sqrtm": {"X": []}}
        return {"sqrtm": {"X": X.tolist()}}