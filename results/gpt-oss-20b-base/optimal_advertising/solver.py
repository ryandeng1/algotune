import numpy as np
from scipy.optimize import linprog

class Solver:
    """
    Fast linear programming solver for the optimal advertising problem.
    The concave objective max sum_i min{R_i * sum_t P_it * D_it, B_i}
    is reformulated as a linear program using auxiliary variables r_i.
    """

    def solve(self, problem: dict) -> dict:
        # Extract parameters
        P = np.array(problem["P"], dtype=np.float64)
        R = np.array(problem["R"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        c = np.array(problem["c"], dtype=np.float64)
        T = np.array(problem["T"], dtype=np.float64)

        m, n = P.shape

        # Decision variables: D (m*n) followed by r (m)
        num_vars = m * n + m

        # Objective: maximize sum r_i  => minimize -sum r_i
        c_obj = np.zeros(num_vars)
        c_obj[m * n :] = -1.0  # r variables

        bounds = [(0, None)] * num_vars  # all variables >= 0

        # Constraints
        A = []  # matrix rows
        b = []

        # Traffic capacity: sum_i D_it <= T_t  (n constraints)
        for t in range(n):
            row = np.zeros(num_vars)
            for i in range(m):
                idx = i * n + t
                row[idx] = 1.0
            A.append(row)
            b.append(T[t])

        # Minimum display: sum_t D_it >= c_i  -> -sum_t D_it <= -c_i
        for i in range(m):
            row = np.zeros(num_vars)
            for t in range(n):
                idx = i * n + t
                row[idx] = -1.0
            A.append(row)
            b.append(-c[i])

        # Revenue upper bounds: r_i <= R_i * sum_t P_it * D_it
        for i in range(m):
            row = np.zeros(num_vars)
            coef = P[i, :] * R[i]
            for t in range(n):
                idx = i * n + t
                row[idx] = -coef[t]  # move to LHS: -coef*D_i_t + r_i <= 0
            row[m * n + i] = 1.0
            A.append(row)
            b.append(0.0)

        # Budget upper bounds: r_i <= B_i
        for i in range(m):
            row = np.zeros(num_vars)
            row[m * n + i] = 1.0
            A.append(row)
            b.append(B[i])

        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64)

        # Solve LP using high‐speed simplex/HUnsc
        res = linprog(
            c_obj,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        status_map = {
            0: "optimal",
            1: "unbounded",
            2: "infeasible",
            3: "unbounded",
            4: "infeasible",
            5: "low_accuracy",
        }

        # Prepare output
        if res.success:
            D_val = res.x[: m * n].reshape((m, n), order="C")
            r_val = res.x[m * n :]

            clicks = np.sum(P * D_val, axis=1)
            revenue_per_ad = np.minimum(R * clicks, B)
            total_revenue = float(np.sum(revenue_per_ad))

            return {
                "status": status_map.get(res.status, "unknown"),
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue_per_ad.tolist(),
                "total_revenue": total_revenue,
                "objective_value": float(-res.fun),  # because we minimized -sum r
            }
        else:
            return {
                "status": status_map.get(res.status, "unknown"),
                "optimal": False,
            }
