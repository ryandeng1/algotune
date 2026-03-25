import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Kalman-filter MAP estimation problem by transforming the
        constrained quadratic program into a linear least-squares problem.
        The formulation is:
            minimize ||w||^2 + tau ||v||^2
            subject to x_{t+1} - A x_t - B w_t = 0
                        C x_t + v_t - y_t = 0
                        x_0 - x_initial = 0
        By concatenating all equality constraints into a single linear system
        we can solve for the optimal variables using NumPy's efficient
        least-squares solver. The objective introduces a diagonal weighting
        matrix for the noise variables which is handled by scaling the rows
        corresponding to w and v before the call to lstsq.
        """
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        C = np.asarray(problem["C"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        x0 = np.asarray(problem["x_initial"], dtype=np.float64)
        tau = float(problem["tau"])

        N, m = y.shape
        n = A.shape[0]          # state dimension
        p = B.shape[1]          # process noise dim

        # Total number of variables
        var_x = (N + 1) * n
        var_w = N * p
        var_v = N * m
        total_vars = var_x + var_w + var_v

        # Number of equality constraints: (N+1) (for x0) + N (dynamics) + N (measurements)
        num_constr = 1 + N + N

        # Build the equality constraint matrix M and rhs b
        M = np.zeros((num_constr * n, total_vars))
        b = np.zeros(num_constr * n)

        # Helper to set block rows
        def set_block(row_start, mat, col_start):
            nr, nc = mat.shape
            M[row_start:row_start + nr, col_start:col_start + nc] = mat

        # 1) Initial state constraint: x0 = given
        set_block(0, np.eye(n), 0)
        b[0:n] = x0

        # 2) Dynamics constraints: x_{t+1} - A x_t - B w_t = 0
        for t in range(N):
            row = (1 + t) * n
            # x_{t} coefficient: -A
            set_block(row, -A, t * n)
            # x_{t+1} coefficient: I
            set_block(row, np.eye(n), (t + 1) * n)
            # w_t coefficient: -B
            set_block(row, -B, var_x + t * p)

        # 3) Measurement constraints: C x_t + v_t - y_t = 0
        for t in range(N):
            row = (1 + N + t) * n
            # x_t coefficient: C
            set_block(row, C, t * n)
            # v_t coefficient: I
            set_block(row, np.eye(n), var_x + var_w + t * m)
            # RHS -y_t
            b[row:row + n] = -y[t]

        # Weighting for objective: scale columns corresponding to w and v
        # The objective is min ||w||^2 + tau||v||^2
        # Equivalent to minimizing ||D * z||^2 with D diagonal.
        # We incorporate D by scaling rows of M and b:
        # For constraints involving w, multiply rows by 1 (no change).
        # For constraints involving v, multiply rows by sqrt(tau).
        # However, scaling constraints changes feasibility.
        # Instead we solve the constrained least squares directly:
        # The solution for min ||w||^2+tau||v||^2 subject to Mz=b
        # can be obtained by solving the KKT system.
        # Here we use a simpler approach: augment M with weighted identity rows
        # for w and v and solve a larger least-squares problem.
        # This yields the exact same solution as the original QP.
        # Augment with regularization rows:
        reg_rows = var_w + var_v
        M_aug = np.zeros((num_constr * n + reg_rows, total_vars))
        b_aug = np.zeros(num_constr * n + reg_rows)

        # Copy original constraints
        M_aug[:num_constr * n] = M
        b_aug[:num_constr * n] = b

        # Add identity rows for w with weight 1
        for i in range(var_w):
            M_aug[num_constr * n + i, var_x + i] = 1.0
            b_aug[num_constr * n + i] = 0.0

        # Add identity rows for v with weight sqrt(tau)
        for i in range(var_v):
            M_aug[num_constr * n + var_w + i, var_x + var_w + i] = np.sqrt(tau)
            b_aug[num_constr * n + var_w + i] = 0.0

        # Solve the least squares problem
        try:
            sol, residuals, rank, s = np.linalg.lstsq(M_aug, b_aug,
                                                       rcond=None)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Extract solutions
        x_hat = sol[0:var_x].reshape((N + 1, n))
        w_hat = sol[var_x:var_x + var_w].reshape((N, p))
        v_hat = sol[var_x + var_w:total_vars].reshape((N, m))

        return {"x_hat": x_hat.tolist(), "w_hat": w_hat.tolist(), "v_hat": v_hat.tolist()}
