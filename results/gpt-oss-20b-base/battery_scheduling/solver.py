from typing import Any, Dict, List
import numpy as np
from scipy.optimize import linprog


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the battery scheduling problem with SciPy's linear programming.
        Returns the same output format as the original CVXPY implementation.
        """
        # --- Parse inputs ----------------------------------------------------
        T = int(problem["T"])
        p = np.asarray(problem["p"], dtype=np.float64)   # prices, shape (T,)
        u = np.asarray(problem["u"], dtype=np.float64)   # base load, shape (T,)
        battery = problem["batteries"][0]               # only one battery

        Q = float(battery["Q"])          # capacity
        C = float(battery["C"])          # max charge rate
        D = float(battery["D"])          # max discharge rate
        η = float(battery["efficiency"])  # efficiency

        n = T                     # number of periods
        m = 3 * n + 1             # # of variables:
        #   q[0..n-1], c_in[0..n-1], c_out[0..n-1], q0 (extra for cycle constraint)

        # --- Variable ordering ----------------------------------------------
        # Index helpers
        idx_q = np.arange(0, n)
        idx_cin = np.arange(n, 2 * n)
        idx_cout = np.arange(2 * n, 3 * n)
        idx_q0 = 3 * n  # last variable: initial charge (restated for cycle)

        # --- Objective ------------------------------------------------------
        # Minimize sum(p[t] * (c_in[t] - c_out[t]))
        c_obj = np.zeros(m)
        c_obj[idx_cin] = p
        c_obj[idx_cout] = -p

        # --- Equality constraints --------------------------------------------
        A_eq = np.zeros((n + 1, m))
        b_eq = np.zeros(n + 1)

        # 1. Battery dynamics for t = 0..n-2
        for t in range(n - 1):
            A_eq[t, idx_q[t]] = -1.0
            A_eq[t, idx_q[t + 1]] = 1.0
            A_eq[t, idx_cin[t]] = 1.0 / η
            A_eq[t, idx_cout[t]] = -η

        # 2. Initial charge equality for cycle constraint
        # q[0] == q[N-1] + η*c_in[N-1] - (1/η)*c_out[N-1]
        A_eq[n, idx_q[0]] = 1.0
        A_eq[n, idx_q[n - 1]] = -1.0
        A_eq[n, idx_cin[n - 1]] = -1.0 / η
        A_eq[n, idx_cout[n - 1]] = η

        # 3. Duplicate initial charge variable (used in dynamics)
        # We equate q[0] with the extra variable q0
        A_eq[n + 1: n + 1, :] = 0
        A_eq[n + 1, idx_q[0]] = -1.0
        A_eq[n + 1, idx_q0] = 1.0
        b_eq[n + 1] = 0.0

        n_eq = n + 1  # ignore last dummy row
        A_eq = A_eq[:n_eq]
        b_eq = b_eq[:n_eq]

        # --- Inequality constraints ------------------------------------------
        A_ub = []
        b_ub = []

        # 1. q bounds: 0 <= q <= Q
        A_ub.append(np.eye(n))
        b_ub.append(np.full(n, Q))
        A_ub.append(-np.eye(n))
        b_ub.append(np.zeros(n))

        # 2. c_in bounds: 0 <= c_in <= C
        A_ub.append(np.eye(n))
        b_ub.append(np.full(n, C))
        A_ub.append(-np.eye(n))
        b_ub.append(np.zeros(n))

        # 3. c_out bounds: 0 <= c_out <= D
        A_ub.append(np.eye(n))
        b_ub.append(np.full(n, D))
        A_ub.append(-np.eye(n))
        b_ub.append(np.zeros(n))

        # 4. No power back to grid: u + c_in - c_out >= 0  ->  -c_in + c_out <= u
        A_ub.append(-np.eye(n))
        b_ub.append(u)

        # Combine all inequalities
        A_ub = np.vstack(A_ub)
        b_ub = np.concatenate(b_ub)

        # --- Solve ----------------------------------------------------------------
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq, bounds=(None, None),
                      method='highs', options={'presolve': True, 'time_limit': 120})

        # --- Build result ----------------------------------------------------------
        if res.success:
            x = res.x
            q_val = x[idx_q].tolist()
            cin_val = x[idx_cin].tolist()
            cout_val = x[idx_cout].tolist()
            c_net = np.array(cin_val) - np.array(cout_val)

            cost_without = float(np.dot(p, u))
            cost_with = float(np.dot(p, u + c_net))
            savings = cost_without - cost_with
            savings_pct = 100.0 * savings / cost_without if cost_without else 0.0

            return {
                "status": "optimal",
                "optimal": True,
                "battery_results": [
                    {
                        "q": q_val,
                        "c": c_net.tolist(),
                        "c_in": cin_val,
                        "c_out": cout_val,
                        "cost": cost_with,
                    }
                ],
                "total_charging": c_net.tolist(),
                "cost_without_battery": cost_without,
                "cost_with_battery": cost_with,
                "savings": savings,
                "savings_percent": savings_pct,
            }
        else:
            return {
                "status": "infeasible" if res.message == "Infeasible" else "solver_error",
                "optimal": False,
                "error": res.message,
            }