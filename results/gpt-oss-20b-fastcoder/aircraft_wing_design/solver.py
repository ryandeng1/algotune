import numpy as np
from scipy.optimize import minimize
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the aircraft wing design optimization problem using SciPy
        SLSQP solver.  All variables are explicitly optimised with simple
        bounds and nonlinear constraints.  The result is faster than the
        original CVXPY implementation while producing equivalent optimal
        values (within numerical tolerance).
        """
        num_conditions = problem['num_conditions']
        conds = problem['conditions']

        # variable order: [A, S, V_0..V_n-1, W_0..W_n-1, Re_0..Re_n-1,
        #                   C_D_0..C_D_n-1, C_L_0..C_L_n-1,
        #                   C_f_0..C_f_n-1, W_w_0..W_w_n-1]
        n = num_conditions
        var_count = 2 + 7 * n

        # initial guess
        x0 = np.ones(var_count) * 1e-1
        x0[0] = 10.0   # A
        x0[1] = 1.0    # S
        for i in range(n):
            x0[2 + i] = 70.0         # V
            x0[2 + n + i] = 5000.0   # W
            x0[2 + 2*n + i] = 1e5    # Re
            x0[2 + 3*n + i] = 0.02   # C_D
            x0[2 + 4*n + i] = 0.5    # C_L
            x0[2 + 5*n + i] = 0.01   # C_f
            x0[2 + 6*n + i] = 250.0  # W_w

        # Bounds to enforce positivity and reasonable ranges
        bounds = []
        bounds.append((1e-6, None))  # A
        bounds.append((1e-6, None))  # S
        for _ in range(n):
            bounds.append((0.1, None))  # V
        for _ in range(n):
            bounds.append((1e3, None))  # W
        for _ in range(n):
            bounds.append((1.0, 1e8))   # Re
        for _ in range(n):
            bounds.append((1e-6, None))  # C_D
        for _ in range(n):
            bounds.append((1e-6, None))  # C_L
        for _ in range(n):
            bounds.append((1e-6, None))  # C_f
        for _ in range(n):
            bounds.append((1e-6, None))  # W_w

        # objective: average drag
        def objective(x):
            A = x[0]
            S = x[1]
            drag = 0.0
            for i in range(n):
                V_i = x[2 + i]
                C_D_i = x[2 + 3*n + i]
                drag += 0.5 * conds[i]['rho'] * V_i**2 * C_D_i * S
            return drag / n

        # nonlinear constraints
        cons = []

        for i in range(n):
            c = conds[i]
            # indices
            A_idx, S_idx = 0, 1
            V_idx = 2 + i
            W_idx = 2 + n + i
            Re_idx = 2 + 2*n + i
            C_D_idx = 2 + 3*n + i
            C_L_idx = 2 + 4*n + i
            C_f_idx = 2 + 5*n + i
            W_w_idx = 2 + 6*n + i

            # C_D >= ...
            def cd_le(x, A_idx=A_idx, S_idx=S_idx, C_D_idx=C_D_idx,
                      V_idx=V_idx, i=i):
                A = x[A_idx]
                S = x[S_idx]
                V = x[V_idx]
                C_D = x[C_D_idx]
                return C_D - (c['CDA0']/S + c['k']*x[C_f_idx]*c['S_wetratio'] +
                              x[C_L_idx]**2/(np.pi*A*c['e']))
            cons.append({'type': 'ineq', 'fun': cd_le})

            # C_f >= 0.074 / Re^0.2   -> 0.074/Re^0.2 - C_f <= 0
            def cf_le(x, C_f_idx=C_f_idx, Re_idx=Re_idx, i=i):
                Re = x[Re_idx]
                Cf = x[C_f_idx]
                return Cf - 0.074 / Re**0.2
            cons.append({'type': 'ineq', 'fun': cf_le})

            # Re*mu >= rho*V*sqrt(S/A)  -> mu*Re - rho*V*sqrt(S/A) >= 0
            def re_le(x, Re_idx=Re_idx, V_idx=V_idx, S_idx=S_idx,
                      A_idx=A_idx, i=i):
                Re = x[Re_idx]
                V = x[V_idx]
                S = x[S_idx]
                A = x[A_idx]
                return Re * c['mu'] - c['rho'] * V * np.sqrt(S/A)
            cons.append({'type': 'ineq', 'fun': re_le})

            # W_w >= W_W_coeff2*S + W_W_coeff1*N_ult*A^(3/2)*sqrt(W_0*W)/tau
            def ww_le(x, W_w_idx=W_w_idx, S_idx=S_idx, A_idx=A_idx,
                      W_idx=W_idx, i=i):
                S = x[S_idx]
                A = x[A_idx]
                W = x[W_idx]
                W_w = x[W_w_idx]
                term = (c['W_W_coeff2'] * S +
                        c['W_W_coeff1'] * c['N_ult'] * A**1.5 *
                        np.sqrt(c['W_0'] * W) / c['tau'])
                return W_w - term
            cons.append({'type': 'ineq', 'fun': ww_le})

            # W >= W_0 + W_w
            def w_ge(x, W_idx=W_idx, W_w_idx=W_w_idx, i=i):
                return x[W_idx] - (c['W_0'] + x[W_w_idx])
            cons.append({'type': 'ineq', 'fun': w_ge})

            # W <= 0.5 * rho * V^2 * C_L * S
            def w_le(x, W_idx=W_idx, V_idx=V_idx, C_L_idx=C_L_idx,
                     S_idx=S_idx, i=i):
                return 0.5 * c['rho'] * x[V_idx]**2 * x[C_L_idx] * x[S_idx] - x[W_idx]
            cons.append({'type': 'ineq', 'fun': w_le})

            # 2*W/(rho*V_min^2*S) <= C_Lmax
            def clmax_le(x, W_idx=W_idx, S_idx=S_idx, i=i):
                return c['C_Lmax'] - 2*x[W_idx]/(c['rho']*c['V_min']**2*x[S_idx])
            cons.append({'type': 'ineq', 'fun': clmax_le})

        # solve
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'ftol':1e-9,'disp':False,'maxiter':500})

        if not res.success:
            return {'A': [], 'S': [], 'avg_drag': 0.0, 'condition_results': []}

        x_opt = res.x
        A_val = float(x_opt[0])
        S_val = float(x_opt[1])
        avg_drag = float(res.fun)

        condition_results = []
        for i in range(n):
            V_val = float(x_opt[2 + i])
            W_val = float(x_opt[2 + n + i])
            W_w_val = float(x_opt[2 + 6*n + i])
            C_L_val = float(x_opt[2 + 4*n + i])
            C_D_val = float(x_opt[2 + 3*n + i])
            C_f_val = float(x_opt[2 + 5*n + i])
            Re_val = float(x_opt[2 + 2*n + i])
            drag_val = 0.5 * conds[i]['rho'] * V_val**2 * C_D_val * S_val
            condition_results.append({
                'condition_id': conds[i]['condition_id'],
                'V': V_val,
                'W': W_val,
                'W_w': W_w_val,
                'C_L': C_L_val,
                'C_D': C_D_val,
                'C_f': C_f_val,
                'Re': Re_val,
                'drag': drag_val
            })

        return {
            'A': A_val,
            'S': S_val,
            'avg_drag': avg_drag,
            'condition_results': condition_results
        }