import numpy as np
from scipy.optimize import linprog

class Solver:

    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using a linear programming
        formulation and scipy's `linprog`.  The problem is:

            maximize   sum_i y_i
            subject to
                y_i <= R[i] * (P[i] @ D[i])
                y_i <= B[i]
                D >= 0
                sum_i D[i, j] <= T[j]   (per time slot)
                sum_j D[i, j] >= c[i]   (per ad )

        The variables are `D` (m * n real) followed by `y` (m real).
        All constraints are linear.  This is a standard LP and can be solved
        very quickly with `scipy.optimize.linprog`.
        """
        P = np.asarray(problem['P'], dtype=float)
        R = np.asarray(problem['R'], dtype=float)
        B = np.asarray(problem['B'], dtype=float)
        c = np.asarray(problem['c'], dtype=float)
        T = np.asarray(problem['T'], dtype=float)

        m, n = P.shape

        # Decision variables: first D (m * n) then y (m)
        num_vars = m * n + m

        # Objective: maximize sum of y_i  -> minimize -sum(y_i)
        c_obj = np.zeros(num_vars)
        c_obj[m * n:] = -1.0

        # Inequality constraints: A_ub x <= b_ub
        # 1. y_i - R[i] * (P[i] @ D[i]) <= 0
        # 2. y_i <= B[i]
        # 3. -D <= 0  (non-negativity)
        # 4. sum_i D[i, j] <= T[j]
        # 5. -D[i, :] <= -c[i]  (sum_j D[i,j] >= c[i])

        A_ub = []
        b_ub = []

        # 1. y_i - R[i] * (P[i] @ D[i]) <= 0
        for i in range(m):
            row = np.zeros(num_vars)
            # coefficient for y_i
            row[m * n + i] = 1.0
            # coefficients for D[i,*]
            idx_start = i * n
            row[idx_start:idx_start + n] = -R[i] * P[i]
            A_ub.append(row)
            b_ub.append(0.0)

        # 2. y_i <= B[i]
        for i in range(m):
            row = np.zeros(num_vars)
            row[m * n + i] = 1.0
            A_ub.append(row)
            b_ub.append(B[i])

        # 3. -D <= 0  (non-negativity)
        for i in range(m):
            for j in range(n):
                row = np.zeros(num_vars)
                idx = i * n + j
                row[idx] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)

        # 4. sum_i D[i, j] <= T[j]
        for j in range(n):
            row = np.zeros(num_vars)
            for i in range(m):
                idx = i * n + j
                row[idx] = 1.0
            A_ub.append(row)
            b_ub.append(T[j])

        # 5. -sum_j D[i,j] <= -c[i]  (i.e., sum_j D[i,j] >= c[i])
        for i in range(m):
            row = np.zeros(num_vars)
            idx_start = i * n
            row[idx_start:idx_start + n] = -1.0
            A_ub.append(row)
            b_ub.append(-c[i])

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # No explicit bounds needed because all are >= 0 and we added above

        res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')

        if res.success:
            x = res.x
            D_val = x[:m * n].reshape((m, n))
            y_val = x[m * n:]

            # Prepare output
            clicks = np.sum(P * D_val, axis=1)
            revenue_per_ad = np.minimum(y_val, B)
            total_revenue = revenue_per_ad.sum()

            return {
                'status': 'optimal',
                'optimal': True,
                'displays': D_val.tolist(),
                'clicks': clicks.tolist(),
                'revenue_per_ad': revenue_per_ad.tolist(),
                'total_revenue': float(total_revenue),
                'objective_value': float(res.fun * -1)  # convert back
            }
        else:
            return {
                'status': res.message,
                'optimal': False,
                'error': str(res)
            }