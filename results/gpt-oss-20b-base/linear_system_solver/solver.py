import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        x = np.linalg.solve(A, b)
        return x.tolist()