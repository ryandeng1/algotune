from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the Chebyshev center problem using SciPy's linear programming solver
        for maximum feasible radius r subject to linear constraints.
        """
        a = np.asarray(problem['a'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64)

        # Pre–compute the norms of the rows of A
        norms = np.linalg.norm(a, axis=1, keepdims=True)

        n = a.shape[1]                      # number of variables in x
        m = a.shape[0]                      # number of inequalities

        # Decision vector: [x_1, …, x_n, r]
        # Objective: maximize r  -> minimize -r
        c = np.zeros(n + 1, dtype=np.float64)
        c[-1] = -1.0

        # Inequality constraints: a_i · x + r * ||a_i|| <= b_i
        A_ub = np.hstack([a, norms])  # shape (m, n+1)
        b_ub = b

        # Bounds: x free, r >= 0
        bounds = [(None, None)] * n + [(0.0, None)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs', options={'presolve': True})

        if result.status != 0:
            raise ValueError(f'Linear program failed: {result.message}')

        sol = result.x[:n]
        return {'solution': sol.tolist()}