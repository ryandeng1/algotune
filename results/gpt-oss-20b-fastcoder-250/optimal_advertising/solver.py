from typing import Any
import numpy as np
from scipy.optimize import linprog


class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using linear programming.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal displays and revenue
        """
        try:
            # Extract problem data
            P = np.array(problem["P"], dtype=float)
            R = np.array(problem["R"], dtype=float)
            B = np.array(problem["B"], dtype=float)
            c = np.array(problem["c"], dtype=float)
            T = np.array(problem["T"], dtype=float)

            m, n = P.shape  # m advertisers, n time slots
            num_vars = m * n + m  # D variables + r variables

            # Objective: maximise sum r_i -> minimise -r
            c_obj = np.zeros(num_vars)
            c_obj[m * n:] = -1.0  # coefficients for r_i

            # Inequality constraints A_ub x <= b_ub
            # 1. traffic capacity per slot: sum_i D_i,j <= T_j
            A_cap = np.zeros((n, num_vars))
            for j in range(n):
                for i in range(m):
                    A_cap[j, i * n + j] = 1.0
            b_cap = T

            # 2. revenue cap per ad: r_i <= B_i
            A_rev_cap = np.zeros((m, num_vars))
            for i in range(m):
                A_rev_cap[i, m * n + i] = 1.0
            b_rev_cap = B

            # 3. linearization: r_i <= R_i * sum_k P_i,k * D_i,k
            A_lin = np.zeros((m, num_vars))
            for i in range(m):
                A_lin[i, m * n + i] = 1.0          # r_i
                A_lin[i, i * n : i * n + n] -= R[i] * P[i]
            b_lin = np.zeros(m)

            # 4. minimum display: sum_i D_i,j >= c_j -> -sum_i D_i,j <= -c_j
            A_min = np.zeros((n, num_vars))
            for j in range(n):
                for i in range(m):
                    A_min[j, i * n + j] = -1.0
            b_min = -c

            # Combine all inequalities
            A_ub = np.vstack([A_cap, A_rev_cap, A_lin, A_min])
            b_ub = np.concatenate([b_cap, b_rev_cap, b_lin, b_min])

            # Bounds: D >= 0, r free but <= ... handled by inequalities
            bounds = [(0, None)] * (m * n) + [(0, None)] * m

            res = linprog(
                c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method="highs",
                options={"presolve": True},
            )

            if not res.success:
                return {"status": res.message, "optimal": False}

            x = res.x
            D_val = x[: m * n].reshape((m, n))
            r_val = x[m * n :].reshape(m)

            # Compute derived metrics
            clicks = np.sum(P * D_val, axis=1)
            revenue_per_ad = np.minimum(R * clicks, B)
            total_revenue = revenue_per_ad.sum()

            return {
                "status": "optimal",
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue_per_ad.tolist(),
                "total_revenue": float(total_revenue),
                "objective_value": float(res.fun * -1),  # because we minimized -sum r
            }

        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}