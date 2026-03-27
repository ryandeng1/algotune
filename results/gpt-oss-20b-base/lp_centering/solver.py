import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the l_p centering problem with a convex quadratic program
        using CVXPY.  The problem is:

                minimize   cᵀx – ∑ log(xᵢ)
                subject to A x = b ,  x > 0

        Parameters
        ----------
        problem : dict
            A dictionary containing the keys ``c``, ``A`` and ``b``.  The
            values are expected to be list-like or ``numpy`` arrays.
            ``c`` is a 1-D array of length *n*, ``A`` is a 2‑D array
            (m×n) and ``b`` is a 1‑D array of length *m*.

        Returns
        -------
        dict
            A dictionary with a single key ``"solution"`` whose value is a
            Python list containing the optimal vector *x*.
        """
        # Convert inputs to NumPy arrays of float64 to avoid dtype
        # conversions inside cvxpy.
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        n = c.size
        x = cp.Variable(n)

        # Construct the objective: cᵀx – ∑ log(xᵢ)
        obj = cp.Minimize(c @ x - cp.sum(cp.log(x)))

        # Equality constraints
        constraints = [A @ x == b]

        # Solve with a fast interior‑point solver
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.CLARABEL)

        # Sanity check and return the solution as a plain Python list.
        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Solver exited with status {prob.status!r}")
        return {"solution": x.value.tolist()}