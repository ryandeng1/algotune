import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Solve the linear system Ax = b using NumPy's fast solver.

        Parameters
        ----------
        problem : dict
            Dictionary containing the matrix `A` and right‑hand side `b`.

        Returns
        -------
        List[float]
            Solution vector as a plain Python list.
        """
        # Convert to numpy arrays without unnecessary copies
        A = np.asarray(problem["A"], dtype=float, order="C")
        b = np.asarray(problem["b"], dtype=float, order="C")

        # Directly solve using the highly optimised LAPACK routine
        x = np.linalg.solve(A, b)

        # Convert result to list only once
        return x.tolist()