import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.array(problem["A"])
        n = A.shape[0]
        eigenvalues = np.linalg.eigvals(A)
        if np.any(np.abs(eigenvalues) >= 1.0 - 1e-8):
            return {"is_stable": False, "P": None}
        
        P = cp.Variable((n, n), symmetric=True)
        constraints = [P >> np.eye(n), A.T @ P @ A - P << -np.eye(n)]
        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return {"is_stable": True, "P": P.value.tolist()}
            else:
                return {"is_stable": False, "P": None}
        except Exception:
            return {"is_stable": False, "P": None}
