import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the battery scheduling problem using SciPy's linear program solver.
        This implementation replaces CVXPY with a fast quadratic‑free LP formulation.
        """
        # ----- Problem data ----------------------------------------------------
        T = int(problem['T'])
        p = np.asarray(problem['p'], dtype=float)          # shape (T,)
        u = np.asarray(problem['u'], dtype=float)          # shape (T,)

        battery = problem['batteries'][0]
        Q = float(battery['Q'])
        C = float(battery['C'])
        D = float(battery['D'])
        eff = float(battery['efficiency'])

        # ----- Variables -------------------------------------------------------
        # x = [q_0 ... q_{T-1}, c_in_0 ... c_in_{T-1}, c_out_0 ... c_out_{T-1}]
        n = 3 * T
        # objective coefficients
        c_obj = np.concatenate([np.zeros(T), p, -p])

        # ----- Constraints ----------------------------------------------------
        A_eq = []
        b_eq = []

        # Balance equations: q[t+1] = q[t] + eff*c_in[t] - (1/eff)*c_out[t]
        for t in range(T - 1):
            row = np.zeros(n)
            row[t] = -1          # -q[t]
            row[t + 1] = 1       # +q[t+1]
            row[T + t] = -eff    # -eff*c_in[t]
            row[T + T + t] = 1/eff  # +(1/eff)*c_out[t]
            A_eq.append(row)
            b_eq.append(0.0)
        # Circular condition: q[0] = q[T-1] + eff*c_in[T-1] - (1/eff)*c_out[T-1]
        row = np.zeros(n)
        row[0] = -1                    # -q[0]
        row[T - 1] = 1                 # +q[T-1]
        row[T + T - 1] = -eff          # -eff*c_in[T-1]
        row[T + T + T - 1] = 1/eff     # +(1/eff)*c_out[T-1]
        A_eq.append(row)
        b_eq.append(0.0)

        A_eq = np.vstack(A_eq)
        b_eq = np.asarray(b_eq)

        # Inequality constraints
        # 1. 0 <= q <= Q
        # 2. 0 <= c_in <= C
        # 3. 0 <= c_out <= D
        # 4. u + c_in - c_out >= 0  -> -(u + c_in - c_out) <= 0  -> -c_in + c_out <= u
        A_ub = []
        b_ub = []

        # Lower bounds are handled by setting lb in linprog directly
        # Upper bounds:
        ub = np.concatenate([np.full(T, Q), np.full(T, C), np.full(T, D)])

        # 4. -c_in + c_out <= u
        row = np.zeros(n)
        row[T:2*T] = -1      # -c_in
        row[2*T:3*T] = 1     # +c_out
        A_ub.append(row)
        b_ub.append(u)

        A_ub = np.vstack(A_ub)
        b_ub = np.concatenate(b_ub)

        # ----- Solve LP -------------------------------------------------------
        res = linprog(c=c_obj,
                      A_eq=A_eq, b_eq=b_eq,
                      A_ub=A_ub, b_ub=b_ub,
                      bounds=(0, None),  # lower bound 0; upper bound handled above
                      method='highs')

        # ----- Build result ----------------------------------------------------
        if res.status not in {0, 1}:  # 0 optimal, 1 optimal with tolerance
            return {'status': 'infeasible', 'optimal': False}

        x = res.x
        q = x[:T]
        c_in = x[T:2*T]
        c_out = x[2*T:3*T]
        c_net = c_in - c_out

        cost_without = float(p @ u)
        cost_with = float(p @ (u + c_net))
        savings = cost_without - cost_with

        return {
            'status': res.message,
            'optimal': True,
            'battery_results': [{
                'q': q.tolist(),
                'c': c_net.tolist(),
                'c_in': c_in.tolist(),
                'c_out': c_out.tolist(),
                'cost': cost_with
            }],
            'total_charging': c_net.tolist(),
            'cost_without_battery': cost_without,
            'cost_with_battery': cost_with,
            'savings': savings,
            'savings_percent': float(100 * savings / cost_without) if cost_without else 0.0
        }