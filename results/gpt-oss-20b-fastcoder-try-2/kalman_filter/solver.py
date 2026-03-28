import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict) -> dict:
        # Extract data
        A = np.asarray(problem['A'])
        B = np.asarray(problem['B'])
        C = np.asarray(problem['C'])
        y = np.asarray(problem['y'])
        x0 = np.asarray(problem['x_initial'])
        tau = float(problem['tau'])
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Indices
        # Unknowns: x[1],...,x[N] (n*N), w[0..N-1] (p*N), v[0..N-1] (m*N)
        var_cnt = n * N + p * N + m * N

        # Build constraint matrix G z = g
        # Rows: dynamics (N*n) + outputs (N*m)
        rows_dyn = N * n
        rows_out = N * m
        G = np.zeros((rows_dyn + rows_out, var_cnt))
        g = np.zeros(rows_dyn + rows_out)

        # Helper to get column offset
        def col_x(t):
            return n * t  # t in 0..N-1 for x[1..N]
        def col_w(t):
            return n * N + p * t
        def col_v(t):
            return n * N + p * N + m * t

        # Dynamics constraints
        for t in range(N):
            # x[t+1] - A x[t] - B w[t] = 0
            # x[t+1] variable
            i_row = t * n
            # coefficient for x[t+1]
            G[i_row:i_row + n, col_x(t)] = np.eye(n)
            # coefficient for -A * x[t]
            if t == 0:
                # x0 known
                g[i_row:i_row + n] = A @ x0
            else:
                G[i_row:i_row + n, col_x(t - 1)] -= A
            # coefficient for -B * w[t]
            G[i_row:i_row + n, col_w(t)] -= B

        # Output constraints
        for t in range(N):
            i_row = rows_dyn + t * m
            # y[t] - C x[t] - v[t] = 0
            g[i_row:i_row + m] = y[t]
            G[i_row:i_row + m, col_x(t)] -= C
            G[i_row:i_row + m, col_v(t)] -= np.eye(m)

        # Objective Hessian H (quadratic term: 1/2 z^T H z)
        # Only w and v enter, with weights 1 and tau
        H = np.zeros((var_cnt, var_cnt))
        # w part
        for t in range(N):
            idx = col_w(t)
            H[idx:idx + p, idx:idx + p] = np.eye(p)
        # v part
        for t in range(N):
            idx = col_v(t)
            H[idx:idx + m, idx:idx + m] = tau * np.eye(m)

        # Solve KKT system: [H  G^T] [z]   = [0]
        #                    [G  0 ] [lam]   = [g]
        K = np.block([
            [H,               G.T],
            [G,               np.zeros((G.shape[0], G.shape[0]))]
        ])
        rhs = np.concatenate([np.zeros(var_cnt), g])

        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            return {'x_hat': [], 'w_hat': [], 'v_hat': []}

        z = sol[:var_cnt]
        # Assemble results
        x_rows = []
        x_rows.append(x0.tolist())
        for t in range(N):
            x_t = z[col_x(t):col_x(t)+n]
            x_rows.append(x_t.tolist())

        w_rows = []
        for t in range(N):
            w_t = z[col_w(t):col_w(t)+p]
            w_rows.append(w_t.tolist())

        v_rows = []
        for t in range(N):
            v_t = z[col_v(t):col_v(t)+m]
            v_rows.append(v_t.tolist())

        return {'x_hat': x_rows, 'w_hat': w_rows, 'v_hat': v_rows}