import numpy as np
import cvxpy as cp

class Solver:
    def __init__(self):
        # default parameters that may be used if not provided in problem dict
        self.beta = 0.95
        self.kappa = 1.0

    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set.

        Parameters
        ----------
        problem : dict
            Dictionary containing the point to project, loss scenarios,
            and parameters (beta, kappa).

        Returns
        -------
        dict
            Dictionary containing the projected point under key "x_proj".
        """
        # Extract problem data
        x0 = np.array(problem["x0"], dtype=float)
        A = np.array(problem["loss_scenarios"], dtype=float)
        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))

        n_scenarios, n_dims = A.shape

        # Decision variable
        x = cp.Variable(n_dims)

        # Objective: minimize squared Euclidean distance to x0
        objective = cp.Minimize(cp.sum_squares(x - x0))

        # CVaR constraint
        k = int(np.ceil((1 - beta) * n_scenarios))
        # If k=0, CVaR constraint is inactive
        constraints = []
        if k > 0:
            constraints.append(cp.sum_largest(A @ x, k) <= kappa * k)

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
                return {"x_proj": []}
            return {"x_proj": x.value.tolist()}
        except (cp.SolverError, Exception):
            return {"x_proj": []}
