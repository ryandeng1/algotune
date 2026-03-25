import numpy as np
from scipy.optimize import linprog


class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using a linear programming formulation.
        The objective is reformulated by introducing auxiliary variables z_i for each ad
        to linearize the piecewise objective:  max Σ min{R_i * clicks_i, B_i}.

        Parameters
        ----------
        problem : dict
            Dictionary containing problem parameters:
            - "m": number of ads
            - "n": number number of time slots
            - "P": click‑through rates (m x n matrix)
            - "R": revenue per click (m‑vector)
            - "B": budget limits (m‑vector)
            - "c": minimum display requirements (m‑vector)
            - "T": traffic capacities (n‑vector)

        Returns
        -------
        dict
            Dictionary containing the optimal solution and related metrics.
        """
        # Extract parameters
        P = np.asarray(problem["P"], dtype=float)
        R = np.asarray(problem["R"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        c = np.asarray(problem["c"], dtype=float)
        T = np.asarray(problem["T"], dtype=float)

        m, n = P.shape

        # Variables: D (m*n) followed by z (m)
        num_vars = m * n + m

        # Objective: maximize sum z_i => minimize -sum z_i
        c_obj = np.zeros(num_vars, dtype=float)
        c_obj[m * n :] = -1.0

        A_ub = []
        b_ub = []

        # Traffic capacity constraints: sum_i D_it <= T_t
        for t in range(n):
            row = np.zeros(num_vars, dtype=float)
            for i in range(m):
                idx = i * n + t
                row[idx] = 1.0
            A_ub.append(row)
            b_ub.append(T[t])

        # Minimum display constraints: sum_t D_it >= c_i  =>  -sum_t D_it <= -c_i
        for i in range(m):
            row = np.zeros(num_vars, dtype=float)
            for t in range(n):
                idx = i * n + t
                row[idx] = -1.0
            A_ub.append(row)
            b_ub.append(-c[i])

        # z_i <= B_i  =>  z_i <= B_i
        for i in range(m):
            row = np.zeros(num_vars, dtype=float)
            idx_z = m * n + i
            row[idx_z] = 1.0
            A_ub.append(row)
            b_ub.append(B[i])

        # z_i <= R_i * (P_i @ D_i)  =>  z_i - R_i * Σ_t P_it D_it <= 0
        for i in range(m):
            row = np.zeros(num_vars, dtype=float)
            idx_z = m * n + i
            row[idx_z] = 1.0
            for t in range(n):
                idx_d = i * n + t
                row[idx_d] = -R[i] * P[i, t]
            A_ub.append(row)
            b_ub.append(0.0)

        A_ub = np.vstack(A_ub)
        b_ub = np.array(b_ub, dtype=float)

        # Variable bounds: all >= 0
        bounds = [(0.0, None) for _ in range(num_vars)]

        # Solve with high‑sensitivity LP solver
        res = linprog(c_obj,
                      A_ub=A_ub,
                      b_ub=b_ub,
                      bounds=bounds,
                      method="highs",
                      options={"presolve": True, "time_limit": 30})

        if res.status != 0:  # 0 indicates success
            return {
                "status": "solver_error",
                "optimal": False,
                "error": f"linprog status {res.status}: {res.message}",
            }

        # Extract solution
        D_opt = res.x[: m * n].reshape((m, n))
        z_opt = res.x[m * n :]

        # Compute metrics
        clicks = np.sum(P * D_opt, axis=1)
        revenue_per_ad = np.minimum(R * clicks, B)
        total_revenue = revenue_per_ad.sum()

        return {
            "status": "optimal",
            "optimal": True,
            "displays": D_opt.tolist(),
            "clicks": clicks.tolist(),
            "revenue_per_ad": revenue_per_ad.tolist(),
            "total_revenue": float(total_revenue),
            "objective_value": float(res.fun * -1),  # since we minimized -sum z
        }
