from typing import Any
import cvxpy as cp
import numpy as np

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
        # Convert inputs to numpy arrays
        x0 = np.asarray(problem['x0'], dtype=float)
        A = np.asarray(problem['loss_scenarios'], dtype=float)

        beta = float(problem.get('beta', self.beta))
        kappa = float(problem.get('kappa', self.kappa))

        n_scenarios, n_dims = A.shape
        # Decision variable
        x = cp.Variable(n_dims)

        # Objective: minimize Euclidean distance to x0
        objective = cp.Minimize(cp.sum_squares(x - x0))

        # CVaR constraint
        k = int(np.floor((1 - beta) * n_scenarios))
        alpha = kappa * k

        # Use sorted values of A @ x to express sum_largest
        Ax = A @ x
        constraints = [cp.sum_largest(Ax, k) <= alpha]

        prob = cp.Problem(objective, constraints)

        try:
            # Solve using a fast convex solver (ECOS)
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=2000, feastol=1e-9)

            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
                return {'x_proj': []}
            return {'x_proj': x.value.tolist()}
        except Exception:
            return {'x_proj': []}