import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict) -> dict:
        # Parameters
        P = np.asarray(problem['P'], dtype=float)
        R = np.asarray(problem['R'], dtype=float)
        B = np.asarray(problem['B'], dtype=float)
        c = np.asarray(problem['c'], dtype=float)
        T = np.asarray(problem['T'], dtype=float)

        m, n = P.shape
        num_vars = m * n + m  # D variables + revenue variables

        # Objective: maximize sum of revenue variables
        c_obj = np.zeros(num_vars)
        c_obj[m * n:] = -1.0  # negative because linprog does minimization

        # Inequality constraints A_ub * x <= b_ub
        # 1. sum of D per time slot <= T
        A_time = np.zeros((n, num_vars))
        for j in range(n):
            for i in range(m):
                A_time[j, i * n + j] = 1.0
        b_time = T.copy()

        # 2. revenue_i <= R_i * P_i·D_i
        A_rev = np.zeros((m, num_vars))
        for i in range(m):
            for col in range(n):
                A_rev[i, i * n + col] = R[i] * P[i, col]
            A_rev[i, m * n + i] = -1.0  # -revenue_i
        b_rev = np.zeros(m)

        # 3. revenue_i <= B_i
        A_b = np.zeros((m, num_vars))
        for i in range(m):
            A_b[i, m * n + i] = -1.0
        b_b = -B.copy()

        # Combine inequalities
        A_ub = np.vstack([A_time, A_rev, A_b])
        b_ub = np.concatenate([b_time, b_rev, b_b])

        # Bounds: D >= 0, revenue free (unbounded below but in objective negative)
        bounds = [(0, None)] * (m * n) + [(None, None)] * m

        # Solve LP
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'time_limit': 10})

        status = 'optimal' if res.success else res.message
        optimal = res.success

        if not optimal:
            return {'status': status, 'optimal': False}

        # Extract solution
        D_val = res.x[:m * n].reshape(m, n)
        revenue = res.x[m * n:]

        # Compute clicks for consistency
        clicks = np.sum(P * D_val, axis=1)

        return {
            'status': status,
            'optimal': True,
            'displays': D_val.tolist(),
            'clicks': clicks.tolist(),
            'revenue_per_ad': revenue.tolist(),
            'total_revenue': float(np.sum(revenue)),
            'objective_value': float(res.fun)
        }