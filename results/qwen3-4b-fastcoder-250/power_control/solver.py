import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        G = np.asarray(problem["G"], float)
        sigma = np.asarray(problem["σ"], float)
        P_min = np.asarray(problem["P_min"], float)
        P_max = np.asarray(problem["P_max"], float)
        S_min = float(problem["S_min"])
        n = G.shape[0]
        
        A_ub = S_min * G
        np.fill_diagonal(A_ub, -G.diagonal())
        
        b_ub = -S_min * sigma
        c = np.ones(n)
        
        bounds = [(P_min[i], P_max[i]) for i in range(n)]
        
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='interior-point'
        )
        
        if not res.success:
            raise ValueError(f"LP solver failed (status={res.status})")
        
        return {
            "P": res.x.tolist(),
            "objective": float(res.fun)
        }
