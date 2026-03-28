from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog

class Solver:

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the lp box problem using SciPy's high-performance linprog solver.

        Parameters
        ----------
        problem : dict
            Contains keys 'c', 'A', and 'b' which define the linear program:
                min   cᵀx
                s.t.  A x ≤ b
                      0 ≤ x ≤ 1

        Returns
        -------
        dict
            {"solution": [x₁, x₂, …, xₙ]}
        """
        c = np.asarray(problem["c"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # Bounds: 0 <= x_i <= 1 for all variables
        bounds = [(0.0, 1.0)] * c.size

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs", options={"presolve": True})

        if not res.success:
            raise RuntimeError(f"Linear program did not converge: {res.message}")

        return {"solution": res.x.tolist()}