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

        # Bounds for variables
        bounds = [(P_min[i], P_max[i]) for i in range(n)]

        # Form inequalities A_ub x <= b_ub
        A_ub = []
        b_ub = []

        # For each constraint:
        # Gii*(1+S_min) P_i - S_min * sum_j Gij P_j >= S_min * sigma_i
        # => -Gii*(1+S_min) P_i + S_min * sum_j Gij P_j <= -S_min * sigma_i
        factor = S_min
        for i in range(n):
            coef = - G[i, i] * (1 + S_min)
            row = np.zeros(n)
            row[i] = coef
            row += factor * G[i, :]
            A_ub.append(row)
            b_ub.append(-factor * sigma[i])

        A_ub = np.vstack(A_ub)
        b_ub = np.asarray(b_ub)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method='highs', options={'presolve': True, 'msg': 0})

        if not res.success:
            raise ValueError(f'Linear solver failed: {res.message}')

        return {'P': res.x.tolist(), 'objective': float(res.fun)}