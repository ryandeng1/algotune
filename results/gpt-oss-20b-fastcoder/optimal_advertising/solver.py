import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict

class Solver:

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        P = np.asarray(problem['P'], dtype=float)   # (m,n)
        R = np.asarray(problem['R'], dtype=float)   # (m,)
        B = np.asarray(problem['B'], dtype=float)   # (m,)
        c = np.asarray(problem['c'], dtype=float)   # (m,)
        T = np.asarray(problem['T'], dtype=float)   # (n,)

        m, n = P.shape

        # Decision variables: D (m*n) followed by y (m)
        num_vars = m * n + m

        # Objective: maximize sum(y)  ->  minimize -sum(y)
        obj = np.zeros(num_vars)
        obj[m * n:] = -1.0  # coefficient for y variables

        # Constraints
        A_ub = []
        b_ub = []

        # 1) sum_i D_ij <= T_j  (column constraints)
        for j in range(n):
            row = np.zeros(num_vars)
            for i in range(m):
                row[i * n + j] = 1.0
            A_ub.append(row)
            b_ub.append(T[j])

        # 2) y_i - R_i * sum_j P_ij * D_ij <= 0  (revenue upper bound)
        for i in range(m):
            row = np.zeros(num_vars)
            row[m * n + i] = 1.0                     # y_i coefficient
            row[i * n : (i + 1) * n] -= R[i] * P[i]   # -R_i * P_i
            A_ub.append(row)
            b_ub.append(0.0)

        # 3) y_i <= B_i
        for i in range(m):
            row = np.zeros(num_vars)
            row[m * n + i] = 1.0
            A_ub.append(row)
            b_ub.append(B[i])

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # 4) sum_j D_ij >= c_i  ->  -sum_j D_ij <= -c_i
        A_lb = []
        b_lb = []
        for i in range(m):
            row = np.zeros(num_vars)
            row[i * n : (i + 1) * n] = -1.0
            A_lb.append(row)
            b_lb.append(-c[i])

        A_lb = np.array(A_lb)
        b_lb = np.array(b_lb)

        # Bounds for variables: all >=0
        bounds = [(0, None)] * num_vars

        # Solve LP
        res = linprog(c=obj, A_ub=A_ub, b_ub=b_ub,
                      A_eq=None, b_eq=None,
                      A_lb=A_lb, b_lb=b_lb,
                      bounds=bounds, method='highs')

        if res.success:
            D_val = res.x[:m * n].reshape((m, n))
            y_val = res.x[m * n:][:]
            clicks = np.sum(P * D_val, axis=1)
            revenue = y_val
            total_revenue = np.sum(revenue)

            return {
                'status': 'OPTIMAL',
                'optimal': True,
                'displays': D_val.tolist(),
                'clicks': clicks.tolist(),
                'revenue_per_ad': revenue.tolist(),
                'total_revenue': float(total_revenue),
                'objective_value': float(-res.fun)  # maximize
            }
        else:
            return {
                'status': res.message,
                'optimal': False,
                'error': str(res.message)
            }