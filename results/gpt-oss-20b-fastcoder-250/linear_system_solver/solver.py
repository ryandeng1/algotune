# solver.py
import numpy as np
from typing import Any, List, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Efficiently solve the linear system Ax = b using NumPy's optimized solver.
        """
        A = np.asarray(problem["A"])
        b = np.asarray(problem["b"])
        # np.linalg.solve handles matrix inversion internally and is fast
        x = np.linalg.solve(A, b)
        return x.tolist()
