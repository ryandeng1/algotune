from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert all inputs to NumPy arrays
        P = np.asarray(problem["P"], dtype=np.float64)
        q = np.asarray(problem["q"], dtype=np.float64)
        G = np.asarray(problem["G"], dtype=np.float64)
        h = np.asarray(problem["h"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        n = P.shape[0]
        # Ensure P is symmetric
        P = (P + P.T) / 2.0

        # Define the optimization variable
        x = cp.Variable(n)

        # Objective: 0.5 * x^T P x + q^T x
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x

        # Constraints
        constraints = []
        if G.size:
            constraints.append(G @ x <= h)
        if A.size:
            constraints.append(A @ x == b)

        # Create and solve the problem using OSQP
        prob = cp.Problem(cp.Minimize(objective), constraints)
        optimal_value = prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=2_000_000,  # allow many iterations if needed
        )

        # Check solver status
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status = {prob.status})")

        return {
            "solution": x.value.tolist(),
            "objective": float(optimal_value),
        }