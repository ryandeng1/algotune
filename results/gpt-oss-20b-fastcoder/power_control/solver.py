import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        G = np.asarray(problem['G'], dtype=float)
        sigma = np.asarray(problem['σ'], dtype=float)
        P_min = np.asarray(problem['P_min'], dtype=float)
        P_max = np.asarray(problem['P_max'], dtype=float)
        S_min = float(problem['S_min'])
        n = G.shape[0]

        # Objective: minimize sum(P)
        c = np.ones(n)

        # Bounds (P_min, P_max)
        bounds = [(P_min[i], P_max[i]) for i in range(n)]

        # Inequality constraints: A_ub x <= b_ub
        # Derived from: Gii*(1+S_min)*P[i] - S_min * sum_j Gij * Pj >= S_min * sigma[i]
        # => -Gii*(1+S_min)*P[i] + S_min * sum_j Gij * Pj <= -S_min * sigma[i]
        A_ub = np.empty((n, n), dtype=float)
        b_ub = -S_min * sigma
        coef = 1.0 + S_min
        for i in range(n):
            # Diagonal term
            A_ub[i, i] = -G[i, i] * coef
            # Off‑diagonal terms
            if n > 1:
                A_ub[i] += G[i] * S_min
        # Solve the LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if not res.success:
            raise ValueError(f'Solver failed (message={res.message})')

        return {'P': res.x.tolist(), 'objective': float(res.fun)}