import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the battery scheduling problem using SciPy's linear programming
        (high-performance, no external dependencies).
        """
        # ------------------------------------------------------------------
        # 1. Parse problem data
        # ------------------------------------------------------------------
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=float)          # price vector
        u = np.asarray(problem["u"], dtype=float)          # base load
        battery = problem["batteries"][0]
        Q = float(battery["Q"])        # Max state of charge
        C = float(battery["C"])        # Max charging power
        D = float(battery["D"])        # Max discharging power
        eff = float(battery["efficiency"])

        # ------------------------------------------------------------------
        # 2. Build linear program --------------------------------------------
        # ------------------------------------------------------------------
        n_vars = 3 * T                         # [q, c_in, c_out]
        idx_q      = slice(0, T)
        idx_c_in   = slice(T, 2 * T)
        idx_c_out  = slice(2 * T, 3 * T)

        # Objective: minimize total cost of net battery transfer
        #   cp = c_in - c_out
        #   cost = p @ cp
        c_obj = np.zeros(n_vars)
        c_obj[idx_c_in] =  p
        c_obj[idx_c_out] = -p

        # Bounds for each variable
        bounds = [(0, Q) for _ in range(T)]          # q
        bounds += [(0, C) for _ in range(T)]          # c_in
        bounds += [(0, D) for _ in range(T)]          # c_out

        # ------------------------------------------------------------------
        # 3. Equality constraints: state transitions
        #    q[t+1] - q[t] - eff*c_in[t] + (1/eff)*c_out[t] = 0
        #    (wrap around) q[0] - q[T-1] - eff*c_in[T-1] + (1/eff)*c_out[T-1] = 0
        # ------------------------------------------------------------------
        A_eq = []
        b_eq = []

        # Transition equations
        for t in range(T - 1):
            row = np.zeros(n_vars)
            row[idx_q[t + 1]] = 1.0
            row[idx_q[t]] = -1.0
            row[idx_c_in[t]] = -eff
            row[idx_c_out[t]] = 1.0 / eff
            A_eq.append(row)
            b_eq.append(0.0)

        # Wrap‑around equation
        row = np.zeros(n_vars)
        row[idx_q[0]] = 1.0
        row[idx_q[T - 1]] = -1.0
        row[idx_c_in[T - 1]] = -eff
        row[idx_c_out[T - 1]] = 1.0 / eff
        A_eq.append(row)
        b_eq.append(0.0)

        # ------------------------------------------------------------------
        # 4. Inequality constraints: ensure that base load is still >= 0
        #    c_in - c_out >= -u    ->   -c_in + c_out <= u
        # ------------------------------------------------------------------
        A_ub = np.zeros((T, n_vars))
        for t in range(T):
            # -c_in[t] + c_out[t] <= u[t]
            A_ub[t, idx_c_in[t]] = -1.0
            A_ub[t, idx_c_out[t]] = 1.0
        b_ub = u.copy()

        # ------------------------------------------------------------------
        # 5. Solve the linear program
        # ------------------------------------------------------------------
        res = linprog(
            c=c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": 200},
        )

        # ------------------------------------------------------------------
        # 6. Build result
        # ------------------------------------------------------------------
        if res.status != 0:  # 0 = optimal
            return {
                "status": "solver_error",
                "optimal": False,
                "error": f"linprog status {res.status} (message: {res.message})",
            }

        q_vals   = res.x[idx_q]
        c_in_vals = res.x[idx_c_in]
        c_out_vals = res.x[idx_c_out]
        c_net = c_in_vals - c_out_vals

        cost_without_battery = float(np.dot(p, u))
        cost_with_battery = float(np.dot(p, u + c_net))
        savings = cost_without_battery - cost_with_battery

        return {
            "status": "optimal",
            "optimal": True,
            "battery_results": [
                {
                    "q": q_vals.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in_vals.tolist(),
                    "c_out": c_out_vals.tolist(),
                    "cost": cost_with_battery,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without_battery,
            "cost_with_battery": cost_with_battery,
            "savings": savings,
            "savings_percent": float(100 * savings / cost_without_battery),
        }