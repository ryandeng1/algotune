from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Chebyshev center problem using SciPy's linear programming.
        """
        a = np.asarray(problem['a'], dtype=float)
        b = np.asarray(problem['b'], dtype=float)
        m, n = a.shape

        # Pre‑compute the norms of each row of a
        row_norms = np.linalg.norm(a, axis=1)

        # Build the coefficient matrix for the inequalities
        # Each row: a[i, :]   row_norms[i]
        A_ub = np.column_stack((a, row_norms))
        b_ub = b

        # Objective: maximize r  => minimize -r
        # Variables order: x[0],...,x[n-1], r
        c = np.zeros(n + 1)
        c[-1] = -1.0   # coefficient for r

        # No bounds on x, but r should be >= 0
        bounds = [(None, None)] * n + [(0, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if not res.success:
            raise RuntimeError(f"Linear program failed: {res.message}")

        return {'solution': res.x[:n].tolist()}