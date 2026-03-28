import numpy as np
import cvxpy as cp

class Solver:
    def __init__(self):
        # Default values (used only if not supplied in the problem dict)
        self.beta = 0.95
        self.kappa = 0.1

    def solve(self, problem: dict) -> dict:
        """
        Compute the projection onto the CVaR constraint set using a very compact
        CVXpy formulation.  The solver is chosen to be OSQP which is fast for
        small–medium sized problems and automatically handles the structure
        of the quadratic objective.
        """
        # Extract data
        x0 = np.asarray(problem["x0"], dtype=np.float64)
        A = np.asarray(problem["loss_scenarios"], dtype=np.float64)
        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))

        n_scenarios, n_dims = A.shape

        # Variables
        x = cp.Variable(n_dims)

        # Objective: Euclidean projection
        objective = cp.Minimize(cp.sum_squares(x - x0))

        # CVaR constraint: sum of the largest k losses must be below alpha
        k = int(np.floor((1.0 - beta) * n_scenarios))
        alpha = kappa * k
        constraints = [cp.sum_largest(A @ x, k) <= alpha]

        # Problem definition
        prob = cp.Problem(objective, constraints)

        # Solve with OSQP (fast for QPs)
        try:
            prob.solve(
                solver=cp.OSQP,
                eps_abs=1e-8,
                eps_rel=1e-8,
                verbose=False,
                warm_start=True,
            )
        except cp.SolverError:
            return {"x_proj": []}
        except Exception:
            return {"x_proj": []}

        # Check for feasibility and return the projection
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_proj": []}

        return {"x_proj": x.value.tolist()}