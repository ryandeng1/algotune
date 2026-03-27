import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict) -> dict:
        # Extract problem data efficiently
        x0 = np.asarray(problem["x0"], dtype=np.double)
        A = np.asarray(problem["loss_scenarios"], dtype=np.double)
        beta = float(problem.get("beta", 0.95))
        kappa = float(problem.get("kappa", 1.0))

        n_scenarios, n_dims = A.shape

        # CVaR constraint parameters
        k = int(np.round((1 - beta) * n_scenarios))
        alpha = kappa * k

        # Variable
        x = cp.Variable(n_dims)

        # Objective: minimize squared Euclidean distance to x0
        objective = cp.Minimize(cp.sum_squares(x - x0))

        # CVaR constraint: sum of largest k losses <= alpha
        constraints = [cp.sum_largest(A @ x, k) <= alpha]

        # Problem definition
        prob = cp.Problem(objective, constraints)

        # Solve with a fast solver (OSQP)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
                return {"x_proj": []}
            return {"x_proj": x.value.tolist()}
        except Exception:
            return {"x_proj": []}