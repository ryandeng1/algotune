from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict) -> dict:
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem.get("beta", self.beta))
        kappa = float(problem.get("kappa", self.kappa))
        n_scenarios, n_dims = A.shape

        x = cp.Variable(n_dims)
        objective = cp.Minimize(cp.sum_squares(x - x0))
        k = int((1 - beta) * n_scenarios)
        alpha = kappa * k
        constraints = [cp.sum_largest(A @ x, k) <= alpha]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and x.value is not None:
                return {"x_proj": x.value.tolist()}
            return {"x_proj": []}
        except cp.SolverError:
            return {"x_proj": []}
        except Exception:
            return {"x_proj": []}