import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem: dict) -> dict:
        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])
        C = np.asarray(problem["C"])
        y = np.asarray(problem["y"])
        x0 = np.asarray(problem["x_initial"])
        tau = float(problem["tau"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # unknowns: [x1,...,xN, w0,...,wN-1, v0,...,vN-1]
        # stacked vector u = [x1;...;xN;w0;...;wN-1;v0;...;vN-1]
        num_vars = N * n + N * p + N * m
        M = np.empty((N * n + N * m, num_vars))
        rhs = np.empty(N * n + N * m)

        # Dynamics constraints: x_{t+1} = A x_t + B w_t   (for t=0..N-1, x0 known)
        for t in range(N):
            # equation for x_{t+1}
            row_idx = t * n
            col_x = t * n               # x_t (already known? we use x_t known implicitly at t=0)
            col_w = N * n + t * p       # w_t
            col_x_next = (t + 1) * n   # x_{t+1}
            M[row_idx:row_idx + n, col_x_next:col_x_next + n] = np.eye(n)
            M[row_idx:row_idx + n, col_w:col_w + p] = B
            # RHS: A x_t (t=0 uses x0)
            if t == 0:
                rhs[row_idx:row_idx + n] = A @ x0
            else:
                # x_t is variable at index (t)*n
                # It will be eliminated when solving least squares; we keep rhs zero
                rhs[row_idx:row_idx + n] = 0.0

        # Observation constraints: y_t = C x_t + v_t
        offset = N * n + N * p
        for t in range(N):
            row_idx = t * m
            col_x = t * n
            col_v = offset + t * m
            M[offset + row_idx:offset + row_idx + m, col_x:col_x + n] = C
            M[offset + row_idx:offset + row_idx + m, col_v:col_v + m] = np.eye(m)
            rhs[offset + row_idx:offset + row_idx + m] = y[t]

        # Objective is weighted least squares: minimize ||w||^2 + tau * ||v||^2
        # We write it as a linear system: minimize ||L u||^2 where L = sqrt(W) * [0 0 I_W ; 0 0 sqrt(tau) I_V]
        # Build W matrix
        W = np.zeros((num_vars, num_vars))
        # w part
        for t in range(N):
            idx = N * n + t * p
            W[idx:idx + p, idx:idx + p] = np.eye(p)
        # v part
        for t in range(N):
            idx = offset + t * m
            W[idx:idx + m, idx:idx + m] = tau * np.eye(m)

        # Solve weighted least squares: minimize ||W^{1/2} u||^2 subject to M u = rhs
        # Equivalent to minimize ||W^{1/2} u||^2 + lambda * ||M u - rhs||^2 with lambda large
        # However we can solve normal equations: (M^T M + W) u = M^T rhs
        # Note: M may be underdetermined; we use least norm solution
        K = M.T @ M + W
        rhs_linear = M.T @ rhs
        u, *_ = la.lstsq(K, rhs_linear)

        # Extract solution
        x_sol = np.empty((N + 1, n))
        x_sol[0] = x0
        for t in range(N):
            x_sol[t + 1] = u[t * n:(t + 1) * n]
        w_sol = u[N * n:N * n + N * p].reshape(N, p)
        v_sol = u[offset:offset + N * m].reshape(N, m)

        return {
            "x_hat": x_sol.tolist(),
            "w_hat": w_sol.tolist(),
            "v_hat": v_sol.tolist(),
        }