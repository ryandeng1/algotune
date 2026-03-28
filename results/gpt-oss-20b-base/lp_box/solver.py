from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve a linear program with box constraints efficiently using SciPy's linprog.
        """
        # Extract problem data
        c = np.asarray(problem['c'], dtype=np.float64)
        A = np.asarray(problem['A'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64)

        # All variables are bounded between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(c.size)]

        # Solve using the HiGHS LP solver (default in SciPy >=1.10)
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        # Ensure a solution was found
        if not res.success:
            raise RuntimeError(f"LP solver failed: {res.message}")

        # Return the solution as a Python list
        return {'solution': res.x.tolist()}