from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    """
    Optimised solver for the LP centering problem.
    Utilises CVXPY with the SCS solver (default backend) for speed.
    """

    def __init__(self) -> None:
        # Pre‑allocate variable name to avoid recreating across calls
        self._x = None

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve a convex optimisation problem of the form:
            min     cᵀx - Σ log(x)
            s.t.    A x = b
        Parameters
        ----------
        problem : dict
            Must contain keys 'c', 'A', 'b'.
        Returns
        -------
        dict
            {"solution": [x₀, ..., x_{n-1}]}
        """
        # Extract problem data as NumPy arrays (fast conversion)
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        n = c.size

        # Re‑use CVXPY variable if shape matches
        if self._x is None or self._x.shape[0] != n:
            self._x = cp.Variable(n)

        # Construct objective: minimise cᵀx - sum(log(x))
        obj = cp.Minimize(c @ self._x - cp.sum(cp.log(self._x)))

        # Equality constraints: A x == b
        constraints = [A @ self._x == b]

        # Solve using SCS (fast default solver for this problem type)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False, max_iters=1000, eps=1e-6)

        # Ensure optimality
        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Solver failed with status '{prob.status}'")

        # Return solution as a plain Python list
        return {"solution": self._x.value.tolist()}