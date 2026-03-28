import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Solve the soft‑margin SVM using CVXPY.
        Returns
        -------
        dict
            beta0 : float
            beta  : list[float]
            optimal_value : float
            missclass_error : float
        """
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float).reshape(-1, 1)
        C = float(problem["C"])

        n, p = X.shape
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        obj = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]
        prob = cp.Problem(obj, constraints)

        try:
            # use SCS with a small max_iters and a reasonable eps
            optimal_value = prob.solve(solver=cp.SCS, eps=1e-5, max_iters=10000, warm_start=True)
        except Exception:
            return None

        if beta.value is None or beta0.value is None:
            return None

        pred = X @ beta.value + beta0.value
        missclass_error = float(np.mean((pred * y < 0)))

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass_error,
        }