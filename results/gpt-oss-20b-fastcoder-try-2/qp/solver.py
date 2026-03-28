from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert all inputs to NumPy float arrays
        P = np.asarray(problem["P"], dtype=float)
        q = np.asarray(problem["q"], dtype=float)
        G = np.asarray(problem["G"], dtype=float)
        h = np.asarray(problem["h"], dtype=float)
        A = np.asarray(problem["A"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        n = P.shape[0]

        # Symmetrize P for numerical stability
        P = (P + P.T) / 2.0

        # Build the QP in CVXPY
        x = cp.Variable(n)
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x
        constraints = [G @ x <= h, A @ x == b]
        prob = cp.Problem(cp.Minimize(objective), constraints)

        # Solve with OSQP for speed and accuracy
        optimal_value = prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-8,
            eps_rel=1e-8,
            verbose=False,
        )

        # Raise an error if the solver failed
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status = {prob.status})")

        return {"solution": x.value.tolist(), "objective": float(optimal_value)}