from typing import Any, Dict, List
import numpy as np
from scipy.optimize import minimize

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the LP centering problem:
            maximize   cᵀx - Σ log(x)
            subject to Ax = b
        using a fast generic optimizer (SLSQP).

        Parameters
        ----------
        problem : dict
            Dictionary with keys "c", "A", "b".

        Returns
        -------
        dict
            Dictionary with key "solution" containing the optimal x.
        """

        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        n = c.size

        # Objective: f(x) = -cᵀx + Σ log(x)
        def f(x):
            return -np.dot(c, x) + np.sum(np.log(x))

        # Gradient of the objective
        def grad(x):
            return -c + 1.0 / x

        # Equality constraint: Ax = b
        def con(x):
            return A @ x - b

        # Jacobian of the constraint
        def con_jac(x):
            return A

        # Initial guess: any positive vector satisfying Ax=b
        # Use smallest-norm solution to A x = b, then perturb to be positive
        try:
            # Solve least squares for minimal norm
            x0, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            x0 = np.ones(n)
        if np.any(x0 <= 0):
            x0 = np.maximum(x0, 1.0)

        constraints = {"type": "eq", "fun": con, "jac": con_jac}
        bounds = [(1e-12, None)] * n  # enforce positivity

        res = minimize(
            f,
            x0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )

        if not res.success:
            # Fall back to cvxpy if SLSQP fails
            import cvxpy as cp
            x_var = cp.Variable(n)
            obj = cp.Minimize(-c @ x_var + cp.sum(cp.log(x_var)))
            prob = cp.Problem(obj, [A @ x_var == b])
            prob.solve(solver="CLARABEL", verbose=False)
            sol = x_var.value
        else:
            sol = res.x

        return {"solution": sol.tolist()}
