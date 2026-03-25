import numpy as np
from scipy.optimize import linprog
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Solve the battery scheduling optimisation using SciPy's linear programming solver.
        The formulation is a linear program:
            minimize   pᵀ (c_in - c_out)
            subject to:
                0 ≤ c_in ≤ C
                0 ≤ c_out ≤ D
                0 ≤ q ≤ Q
                u + c_in - c_out ≥ 0
                q[t+1] = q[t] + η c_in[t] - (1/η) c_out[t]  ∀t < T-1
                q[0]   = q[T‑1] + η c_in[T‑1] - (1/η) c_out[T‑1]
        """
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=float)
        u = np.asarray(problem["u"], dtype=float)
        battery = problem["batteries"][0]

        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        e = float(battery["efficiency"])

        # Variable ordering: [c_in(0..T-1), c_out(0..T-1), q(0..T-1)]
        n_vars = 3 * T

        # Objective: c = c_in - c_out
        c_obj = np.empty(n_vars, dtype=float)
        c_obj[:T] =  p
        c_obj[T:2*T] = -p
        c_obj[2*T:] = 0.0

        # Bounds
        bounds = []
        # c_in bounds
        for _ in range(T):
            bounds.append((0.0, C))
        # c_out bounds
        for _ in range(T):
            bounds.append((0.0, D))
        # q bounds
        for _ in range(T):
            bounds.append((0.0, Q))

        # Inequality constraints: u + c_in - c_out >= 0  =>  -c_in + c_out <= -u
        A_ub = np.zeros((T, n_vars), dtype=float)
        b_ub = -u.copy()
        A_ub[:T, :T] = -np.eye(T)          # -c_in
        A_ub[:T, T:2*T] = np.eye(T)        # +c_out

        # Equality constraints
        # Dynamics and cyclic
        A_eq = np.zeros((T + 1, n_vars), dtype=float)
        b_eq = np.zeros(T + 1, dtype=float)

        # Dynamics for t=0..T-2
        for t in range(T - 1):
            # q[t+1] - q[t] - e*c_in[t] + (1/e)*c_out[t] = 0
            A_eq[t, 2*T + (t + 1)] = 1.0    # q[t+1]
            A_eq[t, 2*T + t] = -1.0          # -q[t]
            A_eq[t, t] = -e                   # -e*c_in[t]
            A_eq[t, T + t] = 1.0 / e          # +(1/e)*c_out[t]
            b_eq[t] = 0.0

        # Cyclic constraint: q[0] - q[T-1] - e*c_in[T-1] + (1/e)*c_out[T-1] = 0
        A_eq[T, 2*T + 0] = 1.0            # q[0]
        A_eq[T, 2*T + (T - 1)] = -1.0     # -q[T-1]
        A_eq[T, T - 1] = -e                # -e*c_in[T-1]
        A_eq[T, T + (T - 1)] = 1.0 / e     # +(1/e)*c_out[T-1]
        b_eq[T] = 0.0

        # Solve LP
        # Use high precision to ensure optimality
        res = linprog(
            c=c_obj,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": 60}
        )

        status = "optimal" if res.success else res.status
        if not res.success:
            return {"status": status, "optimal": False}

        # Extract results
        c_in_val = res.x[:T]
        c_out_val = res.x[T:2*T]
        q_val = res.x[2*T:]

        c_net = c_in_val - c_out_val

        cost_without = float(p @ u)
        cost_with = float(p @ (u + c_net))
        savings = cost_without - cost_with

        return {
            "status": status,
            "optimal": True,
            "battery_results": [
                {
                    "q": q_val.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in_val.tolist(),
                    "c_out": c_out_val.tolist(),
                    "cost": cost_with,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without,
            "cost_with_battery": cost_with,
            "savings": savings,
            "savings_percent": 100.0 * savings / cost_without,
        }
