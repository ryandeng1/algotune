import numpy as np
import cvxpy as cp
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines if a continuous‐time linear system x' = A x
        is asymptotically stable by solving the Lyapunov inequality
        A.T P A - P << -I with P >> I.
        """
        # Extract system matrix
        A = np.asarray(problem["A"], dtype=float)
        n = A.shape[0]

        # Construct CVXPY variables and constraints
        P = cp.Variable((n, n), symmetric=True)
        constr = [P >> np.eye(n), A.T @ P @ A - P << -np.eye(n)]

        # Define a trivial objective
        prob = cp.Problem(cp.Minimize(0), constr)

        # Solve with a fast, on‐screen solver
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
            if prob.status in {"optimal", "optimal_inaccurate"}:
                return {"is_stable": True, "P": P.value.tolist()}
        except Exception:
            pass

        # Fallback if first solver fails or is infeasible
        try:
            prob.solve(solver=cp.CVXOPT, verbose=False)
            if prob.status in {"optimal", "optimal_inaccurate"}:
                return {"is_stable": True, "P": P.value.tolist()}
        except Exception:
            pass

        return {"is_stable": False, "P": None}