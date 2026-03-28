from typing import Any, Dict, List
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve the hard‑margin SVM with CVXPY."""
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
        C = float(problem["C"])

        n, p = X.shape
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        obj = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        cons = [xi >= 0, y * (X @ beta + beta0) >= 1 - xi]

        prob = cp.Problem(obj, cons)
        try:
            opt_val = prob.solve(solver=cp.OSQP)
        except Exception:
            return None

        if beta.value is None or beta0.value is None:
            return None

        pred = X @ beta.value + beta0.value
        missclass = float(np.mean(pred * y < 0))

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.ravel().tolist(),
            "optimal_value": float(opt_val),
            "missclass_error": missclass,
        }