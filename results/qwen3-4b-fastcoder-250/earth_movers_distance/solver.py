import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, list[list[float]]]:
        a = problem["source_weights"]
        b = problem["target_weights"]
        M = problem["cost_matrix"]
        n = len(a)
        m = len(b)
        
        P = cp.Variable((n, m), nonneg=True)
        
        objective = cp.sum(M * P)
        
        row_constraints = [cp.sum(P[i, :]) == a[i] for i in range(n)]
        col_constraints = [cp.sum(P[:, j]) == b[j] for j in range(m)]
        
        problem = cp.Problem(cp.Minimize(objective), row_constraints + col_constraints)
        problem.solve(solver=cp.ECOS)
        
        return {"transport_plan": P.value.tolist()}
