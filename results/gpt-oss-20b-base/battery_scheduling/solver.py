import numpy as np
from scipy.optimize import linprog

class Solver:
    """
    A fast linear‑programming implementation of the battery scheduling problem.
    The problem is formulated as a standard LP and solved with SciPy's
    simplex/OSQP solver, which is orders of magnitude faster than CVXPY for
    the small problems encountered here.
    """

    def solve(self, problem: dict) -> dict:
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=float)          # price vector
        u = np.asarray(problem["u"], dtype=float)          # baseline demand
        battery = problem["batteries"][0]
        Q = float(battery["Q"])                           # storage capacity
        C = float(battery["C"])                           # charge limit
        D = float(battery["D"])                           # discharge limit
        eff = float(battery["efficiency"])                # round‑trip efficiency

        # Decision variables: q_t (state of charge), c_in_t, c_out_t
        # Order: [q0..qT-1, c_in0..c_inT-1, c_out0..c_outT-1]
        n_vars = 3 * T

        # Objective: minimize total cost = p @ (u + c_in - c_out)
        # Since u is constant, we only need the coefficient of (c_in - c_out)
        coef = np.zeros(n_vars)
        coef[T:2*T] =  p           # coefficient for c_in
        coef[2*T:3*T] = -p         # coefficient for c_out

        # Constraints
        A_eq = []     # equality constraints
        b_eq = []

        # State‑of‑charge dynamics
        # q_{t+1} = q_t + eff * c_in_t - c_out_t / eff
        for t in range(T - 1):
            row = np.zeros(n_vars)
            row[t] = -1.0                               # -q_t
            row[t+1] = 1.0                              # +q_{t+1}
            row[T + t] = -eff                           # -eff * c_in_t
            row[2*T + t] = 1.0 / eff                    # + (1/eff) * c_out_t
            A_eq.append(row)
            b_eq.append(0.0)

        # Circular condition for 0 == T-1 + Δq_last
        row = np.zeros(n_vars)
        row[0] = 1.0                                    # q_0
        row[T - 1] = -1.0                               # -q_{T-1}
        row[T + T - 1] = -eff                           # -eff * c_in_{T-1}
        row[2*T + T - 1] = 1.0 / eff                    # + (1/eff) * c_out_{T-1}
        A_eq.append(row)
        b_eq.append(0.0)

        # Ensure non‑negative net demand: u + (c_in - c_out) >= 0
        # This is equivalent to: c_out - c_in <= u
        A_ub = []
        b_ub = []

        for t in range(T):
            row = np.zeros(n_vars)
            row[T + t] = -1.0          # -c_in_t
            row[2*T + t] = 1.0         # +c_out_t
            A_ub.append(row)
            b_ub.append(u[t])

        # Variable bounds
        bounds = [(0, Q)] * T + [(0, C)] * T + [(0, D)] * T

        # Solve LP
        res = linprog(
            c=coef,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )

        if not res.success:
            return {
                "status": res.message,
                "optimal": False,
                "error": res.message,
            }

        # Extract solutions
        q = res.x[:T]
        c_in = res.x[T : 2 * T]
        c_out = res.x[2 * T : 3 * T]
        c_net = c_in - c_out

        cost_without_battery = float(p @ u)
        cost_with_battery = float(p @ (u + c_net))
        savings = cost_without_battery - cost_with_battery

        return {
            "status": "optimal",
            "optimal": True,
            "battery_results": [
                {
                    "q": q.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in.tolist(),
                    "c_out": c_out.tolist(),
                    "cost": float(cost_with_battery),
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without_battery,
            "cost_with_battery": cost_with_battery,
            "savings": savings,
            "savings_percent": float(100 * savings / cost_without_battery),
        }