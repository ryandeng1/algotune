from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        A = problem["A"]
        b = problem["b"]
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        if not isinstance(b, np.ndarray):
            b = np.array(b)
        x = np.linalg.solve(A, b)
        return x.tolist()