from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        A = np.asarray(problem["A"])
        b = np.asarray(problem["b"])
        x = np.linalg.solve(A, b)
        return x.tolist()