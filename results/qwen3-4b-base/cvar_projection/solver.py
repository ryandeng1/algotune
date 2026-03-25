import numpy as np
from scipy.optimize import minimize

class Solver:
    def solve(self, problem: dict) -> dict:
        x0 = np.array(problem["x0"])
        A = np.array(problem["loss_scenarios"])
        beta = float(problem["beta"])
        kappa = float(problem["kappa"])
        n_scenarios, n_dims = A.shape
        k = int((1 - beta) * n_scenarios)
        alpha = kappa * k

        if k <= 0:
            return {"x_proj": x0.tolist()}

        def objective(x):
            return np.sum((x - x0) ** 2)

        def constraint(x):
            losses = A @ x
            partitioned = np.partition(losses, -k)
            return np.sum(partitioned[-k:]) - alpha

        res = minimize(
            objective,
            x0,
            constraints=[{'type': 'ineq', 'fun': constraint}],
            method='trust-constr'
        )

        if res.success:
            return {"x_proj": res.x.tolist()}
        else:
            return {"x_proj": []}
