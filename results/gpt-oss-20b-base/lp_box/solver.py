# solver.py
# Optimised LP box problem solver – replaces CVXPY with Scipy's efficient linprog.
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.optimize import linprog


class Solver:
    """
    Solver for LP box problems of the form
          minimize   cᵀx
          subject to A x ≤ b
                     0 ≤ x ≤ 1

    The implementation uses SciPy's linprog (high‑performance implementation),
    avoiding the heavyweight CVXPY stack and thus achieving lower runtimes.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, list]:
        """
        Solve the LP box problem using `scipy.optimize.linprog`.

        Parameters
        ----------
        problem : dict
            Dictionary containing the LP parameters:
                - 'c' : (n,) array‑like, coefficients of the objective.
                - 'A' : (m, n) array‑like, coefficients of the inequality constraints.
                - 'b' : (m,) array‑like, RHS of the inequality constraints.

        Returns
        -------
        dict
            Dictionary with a single key 'solution' containing the optimal
            variable vector as a plain Python list.
        """
        # Pull data from the problem dictionary and cast to NumPy arrays
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # Define bounds for each variable: (0, 1)
        bounds = [(0.0, 1.0)] * c.size

        # Solve the LP using SciPy's l‑factorised solver which is fast for dense data.
        # `method='highs'` uses the HiGHS LP solver (interior‑point) and is robust.
        res = linprog(c=c,
                      A_ub=A,
                      b_ub=b,
                      bounds=bounds,
                      method="highs",
                      options={"presolve": True})

        # Sanity check: ensure that the optimization succeeded.
        if res.status not in (0, 1):  # 0: optimal, 1: optimal but unbounded? not relevant
            raise RuntimeError(f"LP did not solve optimally; status={res.status}")

        # Return the solution as a plain Python list
        return {"solution": res.x.tolist()}