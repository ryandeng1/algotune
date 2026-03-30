# solver.py
from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    """
    Optimised solver for the lp centering problem.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the LP centering problem using CVXPY.

        This code has been written to minimise overhead:
          * numpy arrays are used directly – no unnecessary copies.
          * The CVXPY problem is constructed once and solved with the
            fast CLARABEL solver.
          * The solution is returned as a plain Python list.

        Parameters
        ----------
        problem : dict
            Keys:

            * 'c' : 1‑D list or array of coefficients.
            * 'A' : 2‑D list or array of constraint coefficients.
            * 'b' : 1‑D list or array of RHS values.

        Returns
        -------
        dict
            {'solution': [x1, x2, …, xn]}
        """
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)

        n = c.size
        x = cp.Variable(n)
        # Objective: cᵗx - Σ log(x)
        objective = cp.Minimize(c.T @ x - cp.sum(cp.log(x)))
        constraints = [A @ x == b]
        prob = cp.Problem(objective, constraints)

        # Solve using CLARABEL – the commercial solver with great performance on this
        # type of convex problem.  The `solver_kwargs` are left default to avoid
        # any extra overhead.
        prob.solve(solver=cp.CLARABEL)

        # CVXPY may return a 2‑D array for 0‑dim cases, so make sure we get a flat list.
        solution = x.value
        if solution.ndim > 1:
            solution = solution.ravel()
        return {"solution": solution.tolist()}