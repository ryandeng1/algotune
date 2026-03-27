import numpy as np
from typing import Dict, List, Any

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the problem described by a dictionary containing NumPy arrays
        and float constants.  The function returns a dictionary mapping the
        variable names to lists of float values representing the final state.

        Parameters
        ----------
        problem : dict[str, np.ndarray | float]
            Dictionary containing problem data.  The keys are strings, the
            values are either NumPy arrays or floats.  It is assumed that
            the dictionary contains exactly the keys that are required to
            compute the final state.

        Returns
        -------
        dict[str, list[float]]
            Mapping from each output variable name to a list of float values,
            representing the solution at the last time step.
        """
        # Extract problem data
        A = problem.get("A")
        B = problem.get("B")
        x0 = problem.get("x0")
        t = problem.get("t")

        # Basic sanity checks
        if A is None or B is None or x0 is None or t is None:
            raise ValueError("Missing required problem data")

        # Ensure all inputs are NumPy arrays of the correct shape
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        x0 = np.asarray(x0, dtype=float)
        t = np.asarray(t, dtype=float)

        # Validate dimensions
        if A.ndim != 2:
            raise ValueError("A must be a 2-D matrix")
        if B.ndim != 2:
            raise ValueError("B must be a 2-D matrix")
        if x0.ndim != 1:
            raise ValueError("x0 must be a 1-D vector")
        if t.ndim != 1:
            raise ValueError("t must be a 1-D vector")

        n = A.shape[0]
        if A.shape[1] != n:
            raise ValueError("A must be square")
        if B.shape[0] != n:
            raise ValueError("B rows must match A rows")
        if x0.size != n:
            raise ValueError("x0 size must match A rows")

        # Compute the solution using an explicit Euler integration
        dt = np.diff(t)
        x = x0.copy()
        for step in range(len(dt)):
            x = x + dt[step] * (A @ x + B @ np.zeros((B.shape[1],), dtype=float))

        # Return the final state as a list of floats
        return {"x_final": x.tolist()}