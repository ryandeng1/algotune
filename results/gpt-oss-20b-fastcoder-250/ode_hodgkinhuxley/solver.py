from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        A minimal, highly‑optimized solver that expects the problem dictionary to
        contain two keys:
          * ``"A"`` : a square ndarray representing the coefficient matrix
          * ``"b"`` : a 1‑D ndarray representing the right‑hand side

        The method returns a dictionary with a single key ``"solution"`` whose
        value is a list of floats representing the solution vector.  If the
        matrix is singular an empty list is returned.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Problem description containing matrices and vectors used by the
            solver.

        Returns
        -------
        dict[str, list[float]]
            Dictionary containing the solution as a list of floats.
        """
        # Retrieve the inputs, using numpy arrays for speed and safety
        A = np.asarray(problem.get("A", np.empty((0, 0))), dtype=float)
        b = np.asarray(problem.get("b", np.empty((0,))), dtype=float)

        # Quick exit if dimensions are incompatible or empty
        if A.size == 0 or b.size == 0 or A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
            return {"solution": []}

        try:
            # Solve the linear system using the most efficient LAPACK routine
            sol = np.linalg.solve(A, b)
            # Convert to Python list for the requested return type
            return {"solution": sol.tolist()}
        except np.linalg.LinAlgError:
            # Singular matrix – return empty list to indicate failure
            return {"solution": []}