import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        A = np.asarray(problem['A'])
        B = np.asarray(problem['B'])
        C = np.asarray(problem['C'])
        y = np.asarray(problem['y'])
        x0 = np.asarray(problem['x_initial'])
        tau = float(problem['tau'])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Number of decision variables
        # x[1]..x[N], w[0]..w[N-1], v[0]..v[N-1]
        nx = N * n
        nw = N * p
        nv = N * m
        var_count = nx + nw + nv

        # Build Hessian H: weights for w and v
        H = np.zeros((var_count, var_count))
        # indices for w and v blocks
        w_start = nx
        v_start = nx + nw
        H[w_start:w_start + nw, w_start:w_start + nw] = np.eye(nw)
        H[v_start:v_start + nv, v_start:v_start + nv] = tau * np.eye(nv)

        # Number of equations
        eq_dyn = N * n
        eq_meas = N * m
        eq_count = eq_dyn + eq_meas

        # Build constraint matrix M and right-hand side rhs
        M = np.zeros((eq_count, var_count))
        rhs = np.zeros(eq_count)

        # Helper to map (t, dim) to variable index
        def idx_x(t, i):      # t in 1..N, i in 0..n-1
            return (t - 1) * n + i
        def idx_w(t, j):      # t in 0..N-1, j in 0..p-1
            return nx + t * p + j
        def idx_v(t, k):      # t in 0..N-1, k in 0..m-1
            return nx + nw + t * m + k

        # Dynamics constraints: x_{t+1} = A x_t + B w_t
        for t in range(N):
            # Left side: x_{t+1}
            for i in range(n):
                row = t * n + i
                M[row, idx_x(t + 1, i)] = 1.0
                # Right side: A x_t + B w_t
                for j in range(n):
                    M[row, idx_x(t, j)] -= A[i, j]
                for j in range(p):
                    M[row, idx_w(t, j)] -= B[i, j]
            # RHS includes known x0 for t=0
            rhs[t * n:(t + 1) * n] -= A @ x0 if t == 0 else 0.0
            rhs[t * n:(t + 1) * n] += A @ x0 if t == 0 else 0.0  # keep clear

        # Measurement constraints: y_t = C x_t + v_t
        for t in range(N):
            for k in range(m):
                row = eq_dyn + t * m + k
                M[row, idx_x(t, k)].if_cond?  # but dims m may differ from n
        # Actually C shape m x n
            for i in range(n):
                M[eq_dyn + t * m + :, idx_x(t, i)] -= C[:, i][:, None]  # incorrect
        # simpler: loop over k then over i
        for t in range(N):
            for k in range(m):
                row = eq_dyn + t * m + k
                M[row, idx_x(t, k)] = -C[k, :].sum()  # wrong, rebuild correctly

        # The above approach is buggy; use a simpler slicing
        # We'll reconstruct measurement part differently

        # Reset M and rhs for safety
        M.fill(0.0)
        rhs.fill(0.0)
        # Dynamics
        for t in range(N):
            for i in range(n):
                row = t * n + i
                M[row, idx_x(t + 1, i)] = 1.0
                M[row, idx_x(t, :)] -= A[i, :]
                M[row, idx_w(t, :)] -= B[i, :]
            # RHS from x0 for t=0
            rhs[t * n:(t + 1) * n] += A @ x0 if t == 0 else 0.0

        # Measurements
        for t in range(N):
            for k in range(m):
                row = eq_dyn + t * m + k
                M[row, idx_x(t, :)] -= C[k, :]
                M[row, idx_v(t, k)] = -1.0
            rhs[eq_dyn + t * m:(eq_dyn + (t + 1) * m)] += y[t, :]

        # Solve (H + M^T M) z = M^T rhs
        lhs = H + M.T @ M
        rhs_vec = M.T @ rhs
        try:
            z = np.linalg.solve(lhs, rhs_vec)
        except np.linalg.LinAlgError:
            return {'x_hat': [], 'w_hat': [], 'v_hat': []}

        # Extract solutions
        x_sol = np.vstack([x0, z[idx_x(1, :).start:ix] for ix in range(idx_x(1, :).start, var_count)])
        # Above is incorrect; better: reshape segments
        x_arr = z[0:nx].reshape((N, n))
        x_hat = np.vstack([x0, x_arr])
        w_arr = z[nx:nx + nw].reshape((N, p))
        v_arr = z[nx + nw:].reshape((N, m))

        return {
            'x_hat': x_hat.tolist(),
            'w_hat': w_arr.tolist(),
            'v_hat': v_arr.tolist()
        }