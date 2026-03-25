from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Chebyshev center problem by linear programming.
        The LP formulation is:
            maximize   r
            subject to a_i^T x + r * ||a_i||_2 <= b_i   for all i
                       r >= 0
        """
        a = np.asarray(problem["a"], dtype=np.float64)  # shape (m, n)
        b = np.asarray(problem["b"], dtype=np.float64)  # shape (m,)

        m, n = a.shape
        norm_a = np.linalg.norm(a, axis=1)  # shape (m,)

        # Variables: [x1, ..., xn, r]
        # Objective: minimize -r
        c = np.zeros(n + 1, dtype=np.float64)
        c[-1] = -1.0

        # Inequality constraints
        # A_ub @ vars <= b
        A_ub = np.hstack([a, norm_a.reshape(-1, 1)])
        # Bounds: x free, r >= 0
        bounds = [(None, None)] * n + [(0, None)]

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": 1.0},
        )

        if res.status not in (0, 1):  # 0: optimal, 1: optimal at infinity
            raise ValueError(f"Linprog failed with status {res.status}")

        center = res.x[:n].tolist()
        return {"solution": center}
