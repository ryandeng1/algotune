from typing import Any, Dict
import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        # Parse inputs
        G = np.asarray(problem["G"], dtype=float)
        sigma = np.asarray(problem["σ"], dtype=float)
        P_min = np.asarray(problem["P_min"], dtype=float)
        P_max = np.asarray(problem["P_max"], dtype=float)
        S_min = float(problem["S_min"])

        n = G.shape[0]
        # Compute M = G - diag(G_ii)
        G_diag = np.diag(G)
        M = G - np.diag(G_diag)

        # Construct inequality matrix A_ub and vector b_ub such that
        # A_ub @ P <= b_ub  (we will negate constraints)
        # Original constraint: Gii P_i - S_min * sum_{k!=i} Gik Pk >= S_min * sigma_i
        # => -(Gii e_i^T - S_min * M_i) P <= -S_min * sigma_i
        A_ub = - (np.diag(G_diag) - S_min * M)
        b_ub = - S_min * sigma

        # Bounds for each variable
        bounds = [(float(P_min[i]), float(P_max[i])) for i in range(n)]

        # Solve LP
        res = linprog(c=np.ones(n), A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")

        if not res.success:
            raise ValueError(f"LP solver failed: {res.message}")

        return {"P": res.x.tolist(), "objective": float(res.fun)}
