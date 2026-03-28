import numpy as np
from typing import Any, List, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the linear system Ax = b using NumPy's optimized solver.

        Parameters
        ----------
        problem : dict
            Dictionary containing the square matrix 'A' and the right hand-side 'b'.

        Returns
        -------
        list[float]
            Solution vector x as a plain Python list.
        """
        # Use np.asarray to avoid making a copy if the input is already a NumPy array
        A = np.asarray(problem["A"])
        b = np.asarray(problem["b"])

        # np.linalg.solve is highly optimized (calls LAPACK) and handles both dense
        # and (via data layout) sparse-like structures efficiently.
        x = np.linalg.solve(A, b)

        # Convert to Python list only once; .tolist() is efficient for small arrays
        return x.tolist()