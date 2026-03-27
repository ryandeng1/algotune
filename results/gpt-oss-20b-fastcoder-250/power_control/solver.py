from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert data to numpy arrays
        G = np.asarray(problem["G"], float)
        sigma = np.asarray(problem["σ"], float)
        P_min = np.asarray(problem["P_min"], float)
        P_max = np.asarray(problem["P_max"], float)
        S_min = float(problem["S_min"])
        n = G.shape[0]

        # Decision variable
        P = cp.Variable(n, nonneg=True)

        # Bounds (vectorised)
        constraints = [P >= P_min, P <= P_max]

        # Precompute diagonal of G as a CVXPY constant
        G_const = cp.Constant(G)
        G_ii = cp.diag(G_const)

        # Interference term: σ + G P - G_ii * P
        # G_ii * P is element‑wise product
        interf = sigma + G_const @ P - G_ii * P

        # Coupling constraints: G_ii * P >= S_min * interf
        constraints.append(G_ii * P >= S_min * interf)

        # Problem definition
        prob = cp.Problem(cp.Minimize(cp.sum(P)), constraints)

        # Solve with ECOS
        prob.solve(solver=cp.ECOS)

        # Check solution status
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status={prob.status})")

        return {"P": P.value.tolist(), "objective": float(prob.value)}