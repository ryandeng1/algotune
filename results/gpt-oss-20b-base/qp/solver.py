from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert data to numpy arrays of type float64
        P = np.asarray(problem['P'], dtype=float)
        q = np.asarray(problem['q'], dtype=float)
        G = np.asarray(problem['G'], dtype=float)
        h = np.asarray(problem['h'], dtype=float)
        A = np.asarray(problem['A'], dtype=float)
        b = np.asarray(problem['b'], dtype=float)

        # Symmetrize P to guarantee numerical PSD-ness
        P = (P + P.T) * 0.5

        n = P.shape[0]
        x = cp.Variable(n)

        # Objective: (1/2)xᵀPx + qᵀx
        objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x

        # Constraints: Gx <= h , Ax == b
        constraints = []
        if G.size:
            constraints.append(G @ x <= h)
        if A.size:
            constraints.append(A @ x == b)

        prob = cp.Problem(cp.Minimize(objective), constraints)

        # Solve using OSQP with tight tolerances
        optimal_value = prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-8,
            eps_rel=1e-8,
            verbose=False,
        )

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f'Solver failed (status = {prob.status})')

        return {
            'solution': x.value.tolist(),
            'objective': float(optimal_value),
        }