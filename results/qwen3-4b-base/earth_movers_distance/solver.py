import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]
        
        n_a = len(a)
        n_b = len(b)
        
        P = cp.Variable((n_a, n_b), nonneg=True)
        objective = cp.sum(M * P)
        constraints = [
            cp.sum(P, axis=1) == a,
            cp.sum(P, axis=0) == b
        ]
        
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.ECOS)
        
        return {"transport_plan": P.value.tolist()}
