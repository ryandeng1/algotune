import numpy as np
import cvxpy as cp

class Solver:
    def __init__(self):
        self.beta = 0.95
        self.kappa = 0.1

    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set.

        :param problem: Dictionary containing the point to project, loss scenarios, and parameters
        :return: Dictionary containing the projected point
        """
        # Convert inputs to NumPy arrays once
        x0 = np.asarray(problem["x0"], dtype=float)
        A = np.asarray(problem["loss_scenarios"], dtype=float)

        # Retrieve beta and kappa, with defaults if missing
        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))

        n_scenarios, n_dims = A.shape
        k = int((1.0 - beta) * n_scenarios)
        alpha = kappa * k

        # Define the variable and objective
        x = cp.Variable(n_dims)
        objective = cp.Minimize(cp.sum_squares(x - x0))

        # Build the CVaR constraint: sum of largest k elements of A*x <= alpha
        constraints = [cp.sum_largest(A @ x, k) <= alpha]

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)

        # Return the result if solved successfully, otherwise return empty list
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and x.value is not None:
            return {"x_proj": x.value.tolist()}
        return {"x_proj": []}