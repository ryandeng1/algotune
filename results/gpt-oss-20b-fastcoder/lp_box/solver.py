from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp box problem using SciPy's linprog for speed.
        """
        c = np.asarray(problem['c'])
        A = np.asarray(problem['A'])
        b = np.asarray(problem['b'])
        n = c.shape[0]

        # Bounds: 0 <= x_i <= 1 for all i
        bounds = [(0, 1)] * n

        # Use the HiGHS solver, which is fast and works for large problems
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        # If the solver failed to find an optimal solution, raise an error
        if res.status not in (0, 1):  # 0: optimal, 1: feasible but unbounded
            raise RuntimeError(f"LP solver failed: {res.message}")

        return {'solution': res.x.tolist()}