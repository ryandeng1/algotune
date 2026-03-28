import numpy as np

def solve(problem: dict) -> dict:
    """
    Solves the linear least‑squares problem

        min  Σ||w_t||^2 + τ Σ||v_t||^2
        s.t.  x_0   = x_init
              x_{t+1} = A x_t + B w_t        (t = 0…N-1)
              y_t    = C x_t + v_t           (t = 0…N-1)

    The problem is reduced to a single large linear least‑squares system
    that is solved with NumPy's `lstsq`.  No external optimisation libraries
    are required, which gives a huge speed‑up compared to using CVXPY.

    Parameters
    ----------
    problem : dict
        Dictionary containing the matrices and vectors described above.
        Keys are 'A', 'B', 'C', 'y', 'x_initial', and 'tau'.

    Returns
    -------
    dict
        Dictionary with keys 'x_hat', 'w_hat', 'v_hat'.  Each value is a list
        (not a NumPy array) because the competition judge expects JSON‑serialisable
        objects.  An empty list indicates that the problem was infeasible or
        an error occurred.
    """
    # ------------------------------------------------------------------
    # 1. Pull data out of the dictionary and cast to NumPy arrays
    # ------------------------------------------------------------------
    A = np.asarray(problem["A"], dtype=np.float64)
    B = np.asarray(problem["B"], dtype=np.float64)
    C = np.asarray(problem["C"], dtype=np.float64)
    y = np.asarray(problem["y"], dtype=np.float64)
    x0 = np.asarray(problem["x_initial"], dtype=np.float64)
    tau = float(problem["tau"])

    N, m = y.shape
    n = A.shape[1]
    p = B.shape[1]

    # Check dimensionality
    if len(x0) != n:
        return {"x_hat": [], "w_hat": [], "v_hat": []}
    if C.shape[1] != n:
        return {"x_hat": [], "w_hat": [], "v_hat": []}

    # ------------------------------------------------------------------
    # 2. Assemble the KKT system  [ 0   Aeq^T ] [x] = ...
    #                               [Aeq  Q  ]
    #  where Q is diagonal with weightings:
    #     weight for each w entry   = 1
    #     weight for each v entry   = τ
    #
    #  Because the number of variables is moderate (typically a few thousand),
    #  we can build a sparse‑like matrix using NumPy and then solve the dense
    #  normal equations.  This is ~O(n^3) only in the number of variables,
    #  but the constants are small and no external solvers are involved.
    # ------------------------------------------------------------------

    # Total number of variables
    nx = (N + 1) * n          # states
    nw = N * p                # controls
    nv = N * m                # measurement noises
    nv_total = nx + nw + nv

    # Construct the equality constraints Aeq * z = b
    #   z = [x_0, x_1, …, x_N, w_0, …, w_{N-1}, v_0, …, v_{N-1}]
    # Each block below adds the corresponding rows.

    # (1) Initial condition:   x_0 == x0
    rows = []
    cols = []
    data = []

    # x_0 block
    rows.append(0)
    cols.append(0)                # x_0 variable index
    data.append(1.0)
    # RHS
    b = [x0.copy()]

    eq_rows = 1

    # (2) Dynamic equations:   x_{t+1} - A x_t - B w_t = 0
    for t in range(N):
        # Solve for row: x_{t+1} - A x_t - B w_t = 0
        # Construct row for each row of n
        for i in range(n):
            # coefficient for x_{t+1, i}
            rows.append(eq_rows)
            cols.append((t + 1) * n + i)
            data.append(1.0)

            # coefficient for x_{t, *}
            # ( - A[*, i] ) -> row i of column i? Actually A * x_t: (A @ x_t)[i] = Σ_j A[i, j] x_t[j]
            for j in range(n):
                rows.append(eq_rows)
                cols.append(t * n + j)
                data.append(-A[i, j])

            # coefficient for w_t
            for j in range(p):
                rows.append(eq_rows)
                cols.append(nx + t * p + j)
                data.append(-B[i, j])

            eq_rows += 1
        # RHS zero
    # (3) Measurement equations: y_t - C x_t - v_t = 0
    for t in range(N):
        for i in range(m):
            rows.append(eq_rows)
            cols.append(t * n + i)                # C has shape m x n, use row i
            # For each variable x_t[j] we need coefficient C[i, j]
            for j in range(n):
                rows.append(eq_rows)
                cols.append(t * n + j)
                data.append(-C[i, j])
            # v_t
            rows.append(eq_rows)
            cols.append(nx + nw + t * m + i)
            data.append(-1.0)
            eq_rows += 1

    b = [x0]  # initial RHS
    # Add zeros for dynamic and measurement constraints
    b += [0.0] * (eq_rows - 1)

    # Build sparse matrix as a COO then convert to dense
    Aeq = np.zeros((eq_rows, nv_total), dtype=np.float64)
    Aeq[np.array(rows), np.array(cols)] = np.array(data)

    b = np.asarray(b, dtype=np.float64)

    # ------------------------------------------------------------------
    # 3. Build the weighted least‑squares normal matrix
    #       [A_eq; sqrt(Q)]   z  =  [b; 0]
    #   Where Q is diagonal: 1 for each w entry, τ for each v entry.
    #
    #   This is equivalent to solving the KKT system:
    #       [A_eq^T A_eq + Q] z = A_eq^T b
    # ------------------------------------------------------------------

    # Weighting matrix Q
    Q = np.zeros((nv_total, nv_total), dtype=np.float64)
    # w coefficients: matrix indices nx to nx+nw-1
    Q[nx:nx + nw, nx:nx + nw] = np.eye(nw, dtype=np.float64)
    # v coefficients:
    Q[nx + nw:, nx + nw:] = tau * np.eye(nv, dtype=np.float64)

    # Normal equations: (A_eq^T A_eq + Q) z = A_eq^T b
    AtA = Aeq.T @ Aeq
    M = AtA + Q
    rhs = Aeq.T @ b

    # Solve the symmetric positive‑definite system
    try:
        z = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return {"x_hat": [], "w_hat": [], "v_hat": []}

    # Extract solutions
    x_hat = z[:nx].reshape(N + 1, n).tolist()
    w_hat = z[nx:nx + nw].reshape(N, p).tolist()
    v_hat = z[nx + nw:].reshape(N, m).tolist()

    return {"x_hat": x_hat, "w_hat": w_hat, "v_hat": v_hat}