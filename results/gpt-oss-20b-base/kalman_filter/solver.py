import numpy as np

def solve(problem: dict) -> dict:
    """
    Solve the following optimization problem

        minimize  Σ‖w[t]‖₂²  + τ Σ‖v[t]‖₂²
        subject to
            x[0]      = x0
            x[t+1]    = A x[t] + B w[t]          (t = 0 … N-1)
            y[t]      = C x[t] + v[t]            (t = 0 … N-1)

    Using only NumPy for maximal speed.
    Returns
        {
            'x_hat': list of state vectors,
            'w_hat': list of control vectors,
            'v_hat': list of noise vectors
        }
    or empty lists if the problem cannot be solved.
    """
    # Extract matrices and data
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
    # x[1..N], w[0..N-1], v[0..N-1]
    n_x = N * n
    n_w = N * p
    n_v = N * m
    n_vars = n_x + n_w + n_v

    # Build the linear equality constraint matrix M and RHS b
    # Each constraint adds one block row of shape (var_shape)
    # We will assemble as a sparse-like dense block matrix
    M = np.empty((N * (n + m), n_vars), dtype=np.float64)
    b = np.empty(N * (n + m), dtype=np.float64)

    # Helper to locate slices
    def idx_x(t):
        # x indices for time t (0..N)
        return slice(t * n, (t + 1) * n)

    def idx_w(t):
        return slice(n_x + t * p, n_x + (t + 1) * p)

    def idx_v(t):
        return slice(n_x + n_w + t * m, n_x + n_w + (t + 1) * m)

    row = 0
    for t in range(N):
        # Dynamics: x[t+1] - A x[t] - B w[t] = 0
        # Shift to RHS the known part when t == 0 (x0 given)
        # Left-hand side over variables
        # x[t+1] coefficient = I
        # x[t] coefficient = -A
        # w[t] coefficient = -B
        # v[t] none
        # If t == 0, move -A*x0 to RHS
        M[row:row + n, idx_x(t + 1)] = np.eye(n)
        M[row:row + n, idx_x(t)] = -A
        M[row:row + n, idx_w(t)] = -B
        b[row:row + n] = np.zeros(n, dtype=np.float64)
        if t == 0:
            b[row:row + n] -= -A @ x0  # add A*x0 to RHS
        row += n

        # Measurement: C x[t] + v[t] - y[t] = 0
        M[row:row + m, idx_x(t)] = C
        M[row:row + m, idx_v(t)] = np.eye(m)
        b[row:row + m] = y[t]
        row += m

    # Build the Hessian H for the quadratic term:
    #   z^T H z = ‖w‖² + τ‖v‖²
    H = np.zeros((n_vars, n_vars), dtype=np.float64)
    # w part: identity
    H[n_x:n_x + n_w, n_x:n_x + n_w] = np.eye(n_w)
    # v part: tau * I
    H[n_x + n_w:n_vars, n_x + n_w:n_vars] = tau * np.eye(n_v)

    # Solve the KKT system:
    #   [ H  M^T ] [z]   = [0]
    #   [ M   0  ] [λ]   = [b]
    # Stack the matrices
    top = np.hstack((H, M.T))     # shape (n_vars, n_vars+eq)
    bottom = np.hstack((M, np.zeros((N * (n + m), N * (n + m)), dtype=np.float64)))
    KKT = np.vstack((top, bottom))

    rhs = np.concatenate((np.zeros(n_vars), b))

    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        return {'x_hat': [], 'w_hat': [], 'v_hat': []}

    z = sol[:n_vars]
    # Extract blocks
    x_sol = z[idx_x(1):idx_x(N + 1)].reshape(N, n)
    w_sol = z[idx_w(0):idx_w(N)].reshape(N, p)
    v_sol = z[idx_v(0):idx_v(N)].reshape(N, m)

    # Construct x vector including initial state
    x_vec = [x0.tolist()] + x_sol.tolist()
    w_vec = w_sol.tolist()
    v_vec = v_sol.tolist()

    return {'x_hat': x_vec, 'w_hat': w_vec, 'v_hat': v_vec}