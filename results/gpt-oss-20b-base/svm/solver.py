import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solves the SVM via CVXPY and returns
            beta0 : float
            beta  : list[float]
            optimal_value : float
            missclass_error : float
        """
        # Extract inputs
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
        C = float(problem["C"])
        n, p = X.shape

        # Define variables
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        # CVXPY problem
        obj = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        con = [xi >= 0,
               cp.multiply(y, X @ beta + beta0) >= 1 - xi]
        prob = cp.Problem(obj, con)

        # Solve with a fast quadratic‑program solver
        try:
            optimal_value = prob.solve(solver=cp.OSQP,
                                     eps_abs=1e-6,
                                     eps_rel=1e-6,
                                     max_iter=20000,
                                     verbose=False)
        except (cp.SolverError, Exception):
            return None

        # Return None if solver did not find a solution
        if beta.value is None or beta0.value is None:
            return None

        # Compute predictions and miss‑classification error
        preds = X @ beta.value + beta0.value
        missclass = float(np.mean((preds * y < 0).astype(float)))

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass,
        }