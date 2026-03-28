from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the Chebyshev center problem using scipy's linear programming.

        :param problem: A dictionary of the Chebyshev center problem's parameters.
        :return: A dictionary with key "solution": a 1D list with n elements representing the solution to the Chebyshev center problem.
        """
        a = np.asarray(problem['a'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64)
        n = a.shape[1]
        # Norm of each row of a
        norm = np.linalg.norm(a, axis=1)

        # Build inequality matrix: A * [x; r] <= b
        # For each row i: a[i] * x + norm[i] * r <= b[i]
        A_ineq = np.hstack([a, norm.reshape(-1, 1)])
        # Objective: maximize r  -> minimize -r
        c = np.zeros(n + 1)
        c[-1] = -1.0

        res = linprog(c, A_ub=A_ineq, b_ub=b, bounds=[(None, None)] * n + [(None, None)],
                      method='highs', options={'presolve': True, 'time_limit': 30})
        if not res.success:
            raise ValueError(f"LP failed: {res.message}")

        x_val = res.x[:-1]  # ignore r
        return {'solution': x_val.tolist()}