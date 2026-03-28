from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        A = problem["matrix"]
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        V = Vh.T
        return {
            "U": U.tolist(),
            "S": s.tolist(),
            "V": V.tolist()
        }