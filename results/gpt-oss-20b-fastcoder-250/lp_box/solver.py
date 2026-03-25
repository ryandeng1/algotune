# solver.py
import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the LP box problem using scipy's linprog for speed.
        The problem is:
            minimize   c^T x
            subject to A x <= b
                       0 <= x <= 1
        Parameters
        ----------
        problem : dict
            Dictionary with keys "c", "A", "b".
        Returns
        -------
        dict
            Dictionary with key "solution" containing the optimal x as a list.
        """
        # Convert inputs to numpy arrays
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # Number of variables
        n = c.shape[0]

        # Bounds for each variable: (0, 1)
        bounds = [(0.0, 1.0)] * n

        # linprog expects a, b_ub for a_ub x <= b_ub
        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
            options={"time_limit": 180}  # limit to 3 minutes if invoked by external tools
        )

        # Ensure optimality
        if not res.success:
            raise RuntimeError(f"Linear program did not converge: {res.message}")

        # Return solution as list
        return {"solution": res.x.tolist()}
