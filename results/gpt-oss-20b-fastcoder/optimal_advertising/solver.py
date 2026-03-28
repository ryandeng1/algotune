import numpy as np
from scipy.optimize import linprog
from typing import Dict, Any

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the optimal advertising problem using SciPy's linear programming.
        """
        P = np.asarray(problem["P"], dtype=float)
        R = np.asarray(problem["R"], dtype=float)
        T = np.asarray(problem["T"], dtype=float)
        c = np.asarray(problem["c"], dtype=float)

        m, n = P.shape

        # Decision variables: D (m*n) followed by y (m)
        num_vars = m * n + m

        # Objective: maximize sum y => minimize -sum y
        c_vec = np.zeros(num_vars)
        c_vec[m * n :] = -1.0  # negative for maximization

        # Constraints
        A_ub = []
        b_ub = []

        # 1. Sum of displays per slot <= T
        for j in range(n):
            row = np.zeros(num_vars)
            for i in range(m):
                row[i * n + j] = 1.0
            A_ub.append(row)
            b_ub.append(T[j])

        # 2. y_i <= R_i * sum_j P_ij D_ij  -->  R_i * sum_j P_ij D_ij - y_i >= 0
        for i in range(m):
            row = np.zeros(num_vars)
            for j in range(n):
                row[i * n + j] = -R[i] * P[i, j]
            row[m * n + i] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)

        # 3. y_i <= B_i  -->  y_i <= B_i
        for i in range(m):
            row = np.zeros(num_vars)
            row[m * n + i] = 1.0
            A_ub.append(row)
            b_ub.append(problem["B"][i])

        # 4. Sum of displays for each ad >= c_i  -->  -sum_j D_ij <= -c_i
        for i in range(m):
            row = np.zeros(num_vars)
            for j in range(n):
                row[i * n + j] = -1.0
            A_ub.append(row)
            b_ub.append(-c[i])

        # Bounds: D >= 0, y >= 0
        bounds = [(0, None)] * num_vars

        res = linprog(
            c=c_vec,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            return {"status": res.message, "optimal": False}

        D_val = res.x[: m * n].reshape((m, n))
        y_val = res.x[m * n :]

        clicks = np.sum(P * D_val, axis=1)
        revenue_per_ad = np.minimum(R * clicks, problem["B"])
        total_revenue = revenue_per_ad.sum()

        return {
            "status": "optimal",
            "optimal": True,
            "displays": D_val.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue_per_ad.tolist(),
            "total_revenue": float(total_revenue),
            "objective_value": float(res.fun * -1),  # convert back
        }