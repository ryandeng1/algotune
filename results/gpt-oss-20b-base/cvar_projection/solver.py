import numpy as np
from scipy.optimize import minimize

class Solver:
    def __init__(self):
        self.beta = 0.95
        self.kappa = 0.1

    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set using a fast SLSQP solver.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'x0': Initial point to project.
                - 'loss_scenarios': Matrix of loss scenarios.
                - optional 'beta' and 'kappa' parameters.

        Returns
        -------
        dict
            Dictionary with key 'x_proj' containing the projected point as a list.
        """
        # Prepare data
        x0 = np.asarray(problem['x0'], dtype=np.float64)
        A = np.asarray(problem['loss_scenarios'], dtype=np.float64)
        beta = float(problem.get('beta', self.beta))
        kappa = float(problem.get('kappa', self.kappa))

        n_scenarios, n_dims = A.shape
        k = int((1 - beta) * n_scenarios)
        alpha = kappa * k

        # Objective: 0.5 * ||x - x0||^2
        def objective(x):
            diff = x - x0
            return 0.5 * np.dot(diff, diff)

        # Constraint: sum of largest k entries of A @ x <= alpha
        def constraint(x):
            y = A.dot(x)
            # efficient partial sort to get k largest
            if k <= 0:
                return -alpha  # always satisfied
            # Use np.partition for O(n) selection
            largest_k = np.partition(y, -k)[-k:]
            return np.sum(largest_k) - alpha

        const = {'type': 'ineq', 'fun': constraint}

        # Run SLSQP
        res = minimize(
            objective,
            x0=x0,
            method='SLSQP',
            constraints=[const],
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 500}
        )

        if not res.success or res.x is None:
            return {'x_proj': []}

        return {'x_proj': res.x.tolist()}