import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[int]] | list[float] | int]) -> dict[str, list[list[float]] | float]:
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]
        
        mask = np.zeros((n, n), dtype=bool)
        mask[inds[:, 0], inds[:, 1]] = True
        
        missing_indices = np.where(~mask)
        missing_indices = np.vstack((missing_indices[0], missing_indices[1])).T
        
        B = cp.Variable((n, n), pos=True)
        
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        
        constraints = [
            cp.prod(B[missing_indices[:, 0], missing_indices[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except cp.SolverError:
            return None
        except Exception:
            return None
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        
        if B.value is None:
            return None
        
        return {
            "B": B.value.tolist(),
            "optimal_value": result
        }
