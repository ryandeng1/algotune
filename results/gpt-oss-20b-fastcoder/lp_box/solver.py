# solver.py
import numpy as np
from scipy.optimize import linprog

class Solver:
    """
    A highly‑optimized solver for LP box problems of the form
    minimize   c^T x
    subject to A x <= b
               0 <= x <= 1
    """
    def solve(self, problem):
        """
        Solve the LP using SciPy's highly‑parallel Highs solver.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                'c' : 1‑D iterable of length n
                'A' : 2‑D iterable (m x n)
                'b' : 1‑D iterable of length m

        Returns
        -------
        dict
            {'solution': [x_0, …, x_{n-1}]}
        """
        c = np.asarray(problem['c'], dtype=np.float64)
        A = np.asarray(problem['A'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64)

        # Bounds 0 <= x_i <= 1 for all variables
        bounds = [(0.0, 1.0)] * c.size

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds,
                      method='highs', options={'presolve': True})

        if not res.success:
            raise RuntimeError(f"LP solver failed: {res.message}")

        # Convert np.array to list for consistency with the original interface
        return {'solution': res.x.tolist()}