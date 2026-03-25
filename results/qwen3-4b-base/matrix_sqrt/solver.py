import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        A = problem["matrix"]
        try:
            X, _ = scipy.linalg.sqrtm(A, disp=False)
        except Exception:
            return {"sqrtm": {"X": []}}
        return {"sqrtm": {"X": X.tolist()}}
