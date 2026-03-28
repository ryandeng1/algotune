#!/usr/bin/env python3
import numpy as np
import cvxpy as cp
from typing import Any, Dict, List

class Solver:
    """
    A small wrapper around CVXPY that solves the
    LP‑centering problem
    ``min   cᵀx − ∑ log(x_i)``  subject to  A x = b.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Parameters
        ----------
        problem : dict
            Must contain keys 'c', 'A', 'b'.

        Returns
        -------
        dict
            {"solution": [x1, x2, …, xn]}
        """
        # Use numpy arrays directly if already supplied
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)

        n = c.size
        x = cp.Variable(n, pos=True)          # enforce x > 0 implicitly
        objective = cp.Minimize(c @ x - cp.sum(cp.log(x)))
        constraints = [A @ x == b]
        prob = cp.Problem(objective, constraints)

        # use ECOS (fast interior‑point for small problems)
        prob.solve(solver=cp.ECOS, eps_abs=1e-8, eps_rel=1e-8, verbose=False)

        if prob.status != "optimal":
            raise RuntimeError(f"Solver failed with status {prob.status}")

        return {"solution": x.value.tolist()}