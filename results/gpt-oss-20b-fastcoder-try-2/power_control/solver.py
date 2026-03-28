import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        G = np.asarray(problem['G'], np.float64)
        sigma = np.asarray(problem['σ'], np.float64)
        P_min = np.asarray(problem['P_min'], np.float64)
        P_max = np.asarray(problem['P_max'], np.float64)
        S_min = float(problem['S_min'])
        n = G.shape[0]

        # Build inequalities A x <= b
        # For each i: -a_i*(1+S_min) P_i + S_min*G[i,:]·P <= -S_min * sigma_i
        a = np.diag(G)                    # a_i
        coeff = -a * (1 + S_min)          # -a_i*(1+S_min)
        rows = np.empty((n, n), dtype=np.float64)

        # Outer loop building rows
        for i in range(n):
            rows[i, i] = coeff[i]
            rows[i] += S_min * G[i]       # S_min * G[i,:]
        b = -S_min * sigma

        # Bounds for each variable
        bounds = [(P_min[i], P_max[i]) for i in range(n)]

        res = linprog(c=np.ones(n), A_ub=rows, b_ub=b,
                      bounds=bounds, method='highs', options={'presolve': True})

        if not res.success:
            raise ValueError(f'LP failed (message={res.message})')

        return {
            'P': res.x.tolist(),
            'objective': float(res.fun)
        }