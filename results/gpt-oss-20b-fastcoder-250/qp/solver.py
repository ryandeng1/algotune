# solver.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        # Convert problem data to NumPy arrays
        P = np.ascontiguousarray(problem["Q"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)
        G = np.ascontiguousarray(problem["G"], dtype=np.float64)
        h = np.asarray(problem["h"], dtype=np.float64)
        A = np.ascontiguousarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        n = P.shape[0]

        # Ensure symmetry of P
        P = (P + P.T) * 0.5

        # Construct CVXPY problem
        x = cp.Variable(n)
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x
        constraints = [G @ x <= h, A @ x == b]
        prob = cp.Problem(cp.Minimize(objective), constraints)

        # Solve using OSQP with tight tolerances
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, verbose=False)

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"QP solver failed, status={prob.status}")

        # Return only the solution vector
        return {"solution": x.value.tolist()}
