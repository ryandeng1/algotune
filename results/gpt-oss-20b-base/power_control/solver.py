import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict

class Solver:
    """
    A fast linear‑programming solver that replaces the original cvxpy formulation.
    The problem is transformed into a standard form:
        minimize   cᵀx
        subject to A_eq x = b_eq
                 A_ub x ≤ b_ub
                 l ≤ x ≤ u
    Here the constraints are all inequalities of the form
        Πi ≥ P_min[i]
        Πi ≤ P_max[i]
        Gii*(1+S_min) * Pi - S_min * Σ_j Gij*Pj ≥ S_min * σi
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        G = np.asarray(problem["G"], dtype=float)
        σ = np.asarray(problem["σ"], dtype=float)
        P_min = np.asarray(problem["P_min"], dtype=float)
        P_max = np.asarray(problem["P_max"], dtype=float)
        S_min = float(problem["S_min"])

        n = G.shape[0]

        # Objective: minimize sum(P)
        c = np.ones(n)

        # Upper bound constraints: Pi ≤ P_max
        A_ub1 = np.eye(n)
        b_ub1 = P_max

        # Lower bound constraints: -Pi ≤ -P_min   (converted to ≤)
        A_ub2 = -np.eye(n)
        b_ub2 = -P_min

        # Interaction constraints:
        # (Gii*(1+S_min)) Pi - S_min*Σ_j Gij Pj ≥ S_min*σi
        # equivalently: - (Gii*(1+S_min)) Pi + S_min*Σ_j Gij Pj ≤ -S_min*σi
        A_ub3 = -S_min * G + np.diag(G.diagonal() * (1 + S_min))
        b_ub3 = -S_min * σ

        # Concatenate all inequalities
        A_ub = np.vstack([A_ub1, A_ub2, A_ub3])
        b_ub = np.hstack([b_ub1, b_ub2, b_ub3])

        # No equality constraints
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            method="highs",
            options={"time_limit": 5.0},
        )

        if not res.success:
            raise RuntimeError(f"Solver failed: {res.message}")

        return {"P": res.x.tolist(), "objective": float(res.fun)}