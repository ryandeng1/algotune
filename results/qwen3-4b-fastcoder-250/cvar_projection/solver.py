import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict) -> dict:
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem["beta"])
        kappa = float(problem["kappa"])
        n_scenarios, n_dims = A.shape
        
        k = int((1 - beta) * n_scenarios)
        alpha = kappa * k
        
        x = cp.Variable(n_dims)
        objective = cp.Minimize(cp.sum_squares(x - x0))
        constraints = [cp.sum_largest(A @ x, k) <= alpha]
        
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return {"x_proj": []}
        
        return {"x_proj": x.value.tolist()}
