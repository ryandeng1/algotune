import numpy as np
from scipy.optimize import linprog
from typing import Any, List, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Chebyshev center problem using SciPy's linear programming routine.

        :param problem: A dictionary containing 'a' (m x n matrix) and 'b' (m-elements vector).
        :return: A dictionary with key "solution" containing the n-dimensional Chebyshev center.
        """
        a = np.asarray(problem["a"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)
        m, n = a.shape

        # Norms of each row of a
        norms = np.linalg.norm(a, axis=1)

        # Build the LP: maximize r  <=>  minimize -r
        c = np.concatenate([np.zeros(n), [-1.0]])  # objective coefficients

        # A_ub x + A_ub_r * r <= b  ->  [a | norms] * [x; r] <= b
        A_ub = np.hstack([a, norms[:, None]])  # shape (m, n+1)
        b_ub = b

        # Bounds: x free, r >= 0
        bounds = [(None, None)] * n + [(0, None)]

        # Solve
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method="highs", options={"presolve": True, "time_limit": 10.0})

        if not res.success:
            raise RuntimeError(f"Linear program failed: {res.message}")

        # Extract solution (only x components)
        return {"solution": res.x[:n].tolist()}