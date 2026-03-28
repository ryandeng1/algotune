from typing import Any
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        G = np.asarray(problem["G"], dtype=float)
        σ = np.asarray(problem["σ"], dtype=float)
        P_min = np.asarray(problem["P_min"], dtype=float)
        P_max = np.asarray(problem["P_max"], dtype=float)
        S_min = float(problem["S_min"])

        n = G.shape[0]
        # objective: minimize sum(P)
        c = np.ones(n)

        # Bounds: P_min <= P <= P_max
        bounds = list(zip(P_min, P_max))

        # Inequality constraints: (S_min * sum_j G[i,j] P[j] - (1+S_min)*G[i,i] P[i]) <= -S_min * σ[i]
        A = np.empty((n, n), dtype=float)
        b = -S_min * σ
        for i in range(n):
            # base row: S_min * G[i, :]
            row = S_min * G[i, :].copy()
            # adjust i-th coefficient
            row[i] -= (1 + S_min) * G[i, i]
            A[i, :] = row

        res = linprog(
            c, A_ub=A, b_ub=b, bounds=bounds, method="highs", options={"disp": False}
        )

        if not res.success:
            raise ValueError(f"Solver failed: {res.message}")

        P_opt = res.x
        return {"P": P_opt.tolist(), "objective": float(res.fun)}