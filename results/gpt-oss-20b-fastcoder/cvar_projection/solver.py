import numpy as np
import cvxpy as cp
from typing import Dict, Any

class Solver:
    """Project a point onto a CVaR constraint set."""
    def __init__(self):
        self.beta = 0.95
        self.kappa = 0.1

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the projection onto the CVaR constraint set."""
        # Convert inputs to numpy arrays (float64)
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        A = np.asarray(problem["loss_scenarios"], dtype=np.float64)

        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))

        n_scenarios, n_dims = A.shape
        # Variable to optimize
        x = cp.Variable(n_dims, dtype=cp.float64)

        # Objective: minimise Euclidean distance to x0
        objective = cp.Minimize(cp.sum_squares(x - x0))

        k = int((1 - beta) * n_scenarios)
        if k < 1:                       # CVaR defined only if k>=1
            return {"x_proj": []}
        alpha = kappa * k

        # CVaR constraint: sum of the k largest elements of A @ x <= alpha
        constraints = [cp.sum_largest(A @ x, k) <= alpha]

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)
        except cp.SolverError:
            return {"x_proj": []}

        # Check solver status
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_proj": []}

        return {"x_proj": x.value.tolist()}