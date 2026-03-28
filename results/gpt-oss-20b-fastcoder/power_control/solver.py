import numpy as np
from scipy.optimize import linprog
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        G = np.asarray(problem["G"], dtype=float)
        σ = np.asarray(problem["σ"], dtype=float)
        P_min = np.asarray(problem["P_min"], dtype=float)
        P_max = np.asarray(problem["P_max"], dtype=float)
        S_min = float(problem["S_min"])

        n = G.shape[0]
        # Objective: minimize sum(P)
        c = np.ones(n)

        # Bounds for each variable
        bounds = [(P_min[i], P_max[i]) for i in range(n)]

        # Inequality constraints: A_ub x <= b_ub
        # Derived from (1+S_min)*Gii*Pi - S_min*sum_j Gij*Pj >= S_min*σi
        # => -(1+S_min)*Gii*Pi + S_min*sum_j Gij*Pj <= -S_min*σi
        A_ub = np.zeros((n, n))
        b_ub = -S_min * σ

        for i in range(n):
            A_ub[i, :] = S_min * G[i, :]
            A_ub[i, i] -= (1.0 + S_min) * G[i, i]

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        if res.status != 0:
            raise ValueError(f"Solver failed (status={res.message})")

        return {"P": res.x.tolist(), "objective": float(res.fun)}