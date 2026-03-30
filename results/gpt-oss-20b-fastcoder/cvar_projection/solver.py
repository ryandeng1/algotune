# solver.py
import numpy as np
import cvxpy as cp

class Solver:
    """
    Efficient CVaR projection solver.

    The projection problem is equivalent to a convex quadratic program:

        minimize    ||x - x0||^2
        subject to  sum_largest(A x, k) <= alpha

    where k = floor((1‑beta)·n_scenarios) and α = κ·k.
    We use cvxpy with the OSQP solver, which is fast for this type of problem.
    """

    def __init__(self):
        self.beta = 0.95
        self.kappa = 0.1

    def solve(self, problem: dict) -> dict:
        # Extract data and cast to float64 for performance
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        A = np.asarray(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))

        n_scenarios, n_dims = A.shape
        k = int((1.0 - beta) * n_scenarios)
        if k <= 0:
            # No CVaR constraint – simply return x0
            return {"x_proj": x0.tolist()}

        alpha = kappa * k

        # Create cvxpy variable and problem
        x = cp.Variable(n_dims)

        # Objective: 0.5*||x - x0||²  (factor 0.5 does not change solution)
        objective = cp.Minimize(cp.sum_squares(x - x0))
        constraints = [cp.sum_largest(A @ x, k) <= alpha]
        prob = cp.Problem(objective, constraints)

        # Solve with OSQP (fast for quadratic programs)
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=2000)

        # Return solution if found
        if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
            return {"x_proj": x.value.astype(np.float64).tolist()}

        # If infeasible or error fall back to trivial projection
        return {"x_proj": []}