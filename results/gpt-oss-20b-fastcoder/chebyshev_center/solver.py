import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Fast Chebyshev center solver using SciPy's linear programming (Highs).
        """
        a = np.asarray(problem['a'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64)

        # Compute the row norms of A
        row_norms = np.linalg.norm(a, axis=1)

        n = a.shape[1]          # number of decision variables
        m = a.shape[0]          # number of constraints

        # Variables: [x1,...,xn, r]
        # Objective: maximize r  -> minimize -r
        c = np.zeros(n + 1)
        c[-1] = -1.0   # coefficient for -r

        # Constraints: A x + r * row_norms <= b
        A_ub = np.column_stack([a, row_norms[:, None]])
        b_ub = b

        # Set bounds: x free, r unrestricted (defaults to (-inf, inf))
        bounds = [(None, None)] * n + [(None, None)]

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'time_limit': 10}
        )

        if not res.success:
            raise ValueError("LP did not converge")

        # Extract solution for x
        x_val = res.x[:n]
        return {'solution': x_val.tolist()}