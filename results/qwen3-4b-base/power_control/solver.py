from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        G = np.asarray(problem["G"], float)
        σ = np.asarray(problem["σ"], float)
        P_min = np.asarray(problem["P_min"], float)
        P_max = np.asarray(problem["P_max"], float)
        S_min = float(problem["S_min"])
        n = G.shape[0]

        P = cp.Variable(n, nonneg=True)
        constraints = [P >= P_min, P <= P_max]

        diag = np.diag(G)
        A = (1 + S_min) * np.diag(diag) - S_min * G
        constraints.append(cp.matmul(A, P) >= S_min * σ)

        prob = cp.Problem(cp.Minimize(cp.sum(P)), constraints)
        prob.solve(solver=cp.ECOS)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Solver failed (status={prob.status})")

        return {"P": P.value.tolist(), "objective": float(prob.value)}