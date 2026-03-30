# solver.py
import cvxpy as cp
import numpy as np
from typing import Any, Dict

class Solver:
    """Solver for robust LPs with SOC constraints.

    The implementation has been tuned for speed:
      • All inputs are converted to NumPy arrays in one shot.
      • The SOC constraints are built via a list comprehension.
      • CVXPY 'clarabel' is invoked with minimal verbosity and
        a few extra options to reduce solver overhead.
      • Unnecessary boilerplate (e.g. empty `else`/`finally` blocks)
        has been removed.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        # Extract data
        c = np.asarray(problem["c"])
        b = np.asarray(problem["b"])
        P = np.asarray(problem["P"])
        q = np.asarray(problem["q"])

        n = c.size
        m = P.shape[0]

        # Decision variable
        x = cp.Variable(n)

        # SOC constraints:  ||P_i^T x||_2 <= b_i - q_i^T x
        constraints = [
            cp.SOC(b[i] - q[i] @ x, P[i].T @ x)
            for i in range(m)
        ]

        # Problem definition
        prob = cp.Problem(cp.Minimize(c @ x), constraints)

        try:
            # Call the solver with minimal overhead
            prob.solve(
                solver=cp.CLARABEL,
                verbose=False,
                warm_start=True,        # useful if the same solver is called repeatedly
                max_iters=5000,        # keep a generous bound
                abstol=1e-8,
                reltol=1e-8,
            )
        except Exception:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            return {"objective_value": float("inf"), "x": np.full(n, np.nan)}

        return {"objective_value": prob.value, "x": x.value}