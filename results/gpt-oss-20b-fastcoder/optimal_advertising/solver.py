import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using SciPy LP solver.
        """
        P = np.asarray(problem['P'])
        R = np.asarray(problem['R'])
        B = np.asarray(problem['B'])
        c = np.asarray(problem['c'])
        T = np.asarray(problem['T'])

        m, n = P.shape
        # decision variables: D[i, j] for each ad i and slot j
        # plus rev[i] for revenue of each ad
        # variable index mapping: D(i,j) -> i*n + j, rev(i) -> m*n + i
        num_vars = m * n + m

        # objective: maximize sum rev[i] -> minimize -sum rev[i]
        c_obj = np.zeros(num_vars)
        c_obj[m * n :] = -1.0  # maximize rev

        # bounds: all variables >= 0
        bounds = [(0, None)] * num_vars

        # constraints
        A = []
        b = []

        # rev_i <= R_i * sum_j P[i,j] * D[i,j]
        for i in range(m):
            row = np.zeros(num_vars)
            # coefficient for rev_i
            row[m * n + i] = -1.0
            # coefficients for D[i,j]
            for j in range(n):
                row[i * n + j] = P[i, j] * R[i]
            A.append(row)
            b.append(0.0)

        # rev_i <= B_i
        for i in range(m):
            row = np.zeros(num_vars)
            row[m * n + i] = 1.0
            A.append(row)
            b.append(B[i])

        # sum_i D[i, j] <= T_j
        for j in range(n):
            row = np.zeros(num_vars)
            for i in range(m):
                row[i * n + j] = 1.0
            A.append(row)
            b.append(T[j])

        # sum_j D[i, j] >= c_i -> -sum_j D[i,j] <= -c_i
        for i in range(m):
            row = np.zeros(num_vars)
            for j in range(n):
                row[i * n + j] = -1.0
            A.append(row)
            b.append(-c[i])

        A = np.array(A)
        b = np.array(b)

        # solve LP
        res = linprog(c=c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if not res.success:
            return {'status': res.message, 'optimal': False}

        x = res.x
        D_val = x[:m * n].reshape(m, n)
        rev = x[m * n:]

        clicks = np.sum(P * D_val, axis=1)
        revenue_per_ad = rev
        total_revenue = np.sum(revenue_per_ad)

        return {
            'status': 'optimal',
            'optimal': True,
            'displays': D_val.tolist(),
            'clicks': clicks.tolist(),
            'revenue_per_ad': revenue_per_ad.tolist(),
            'total_revenue': float(total_revenue),
            'objective_value': float(-res.fun)
        }