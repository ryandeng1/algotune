from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog

def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    G = np.asarray(problem["G"], dtype=np.float64)
    sigma = np.asarray(problem["σ"], dtype=np.float64)
    P_min = np.asarray(problem["P_min"], dtype=np.float64)
    P_max = np.asarray(problem["P_max"], dtype=np.float64)
    S_min = float(problem["S_min"])

    n = G.shape[0]

    # Objective: minimize sum(P)
    c = np.ones(n)

    # Bounds: P_min <= P <= P_max
    bounds = [(P_min[i], P_max[i]) for i in range(n)]

    # Interference constraints transformed to linear inequalities
    # Gii(1 - S_min) Pi - S_min * sum_j Gij Pj >= S_min * sigma_i
    # Convert to A_ub x <= b_ub: -(Gii(1 - S_min)) Pi + S_min * sum_j Gij Pj <= -S_min * sigma_i
    A_ub = np.zeros((n, n), dtype=np.float64)
    b_ub = -S_min * sigma

    for i in range(n):
        Gii = G[i, i]
        A_ub[i, i] = -Gii * (1 - S_min)
        A_ub[i] += S_min * G[i]  # adds S_min * sum_j Gij Pj

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success:
        raise ValueError(f"Solver failed: {res.message}")

    return {"P": res.x.tolist(), "objective": float(res.fun)}