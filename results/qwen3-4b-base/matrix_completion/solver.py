import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]
        
        observed_set = set(tuple(pair) for pair in inds)
        all_indices = np.array(list(zip(np.arange(n), np.arange(n))), dtype=int)
        mask = np.array([(i, j) not in observed_set for i, j in all_indices], dtype=bool)
        otherinds = all_indices[mask]
        
        B = cp.Variable((n, n), pos=True)
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except (cp.SolverError, Exception):
            return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        
        B_val = B.value
        if B_val is None:
            return None
        
        return {
            "B": B_val.tolist(),
            "optimal_value": result
        }
