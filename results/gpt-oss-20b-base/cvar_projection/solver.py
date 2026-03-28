import numpy as np
import cvxpy as cp

class Solver:
    beta: float | None = None
    kappa: float | None = None

    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set.

        :param problem: Dictionary containing 'x0', 'loss_scenarios', 'beta', 'kappa'.
        :return: Dictionary with the projected point ('x_proj').
        """
        # Extract data
        x0 = np.asarray(problem["x0"], dtype=float)
        A = np.asarray(problem["loss_scenarios"], dtype=float)
        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))

        n_scenarios, n_dims = A.shape
        k = int(np.floor((1 - beta) * n_scenarios))
        if k <= 0:
            return {"x_proj": x0.tolist()}

        alpha = kappa * k

        # CVaR constraint: sum of largest k elements of A*x <= alpha
        x = cp.Variable(n_dims)
        objective = cp.Minimize(cp.sum_squares(x - x0))
        constraint = cp.sum_largest(A @ x, k) <= alpha

        prob = cp.Problem(objective, [constraint])
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6)

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_proj": []}

        return {"x_proj": x.value.tolist()}