import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """Solve a linear system `Ax = b` using NumPy's fast solver."""
        A = np.array(problem["A"], dtype=np.float64, copy=False)
        b = np.array(problem["b"], dtype=np.float64, copy=False)
        # numpy.linalg.solve is already highly optimized and handles
        # normal and singular systems efficiently.
        return np.linalg.solve(A, b).tolist()