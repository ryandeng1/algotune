# solver.py
import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[float]:
        """
        Solve the linear system Ax = b using NumPy's optimized solver.

        Parameters
        ----------
        problem : dict
            Dictionary with keys "A" and "b". "A" is a square matrix (list of lists),
            "b" is a vector (list).

        Returns
        -------
        list[float]
            Solution vector x such that Ax = b.
        """
        # Convert inputs to NumPy arrays (dtype float for performance)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)

        # Solve the linear system (no explicit inverse)
        x = np.linalg.solve(A, b)

        # Return as a plain Python list
        return x.tolist()
