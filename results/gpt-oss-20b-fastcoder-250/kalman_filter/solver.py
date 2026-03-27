# optimized solution without CVXPY
import numpy as np
from typing import Any

def solve(problem: dict) -> dict:
    # extract data
    A = np.asarray(problem["A"])
    B = np.asarray(problem["B"])
    C = np.asarray(problem["C"])
    y = np.asarray(problem["y"])
    x0 = np.asarray(problem["x_initial"])
    tau = float(problem["tau"])

    N, m = y.shape
    n = A.shape[1]
    p = B.shape[1]

    # Build large linear system
    # Variables order: w[0:N-1], v[0:N-1], x[1:N] (x[0] known)
    # equations:
    # 1. dynamics: x[t+1] = A @ x[t] + B @ w[t]
    #    for t=0..N-1  (N equations, each n rows)
    # 2. measurements: y[t] = C @ x[t] + v[t]
    #    for t=0..N-1  (N equations, each m rows)

    # Unknown vector size
    w_block_size = N * p
    v_block_size = N * m
    x_block_size = N * n
    K = w_block_size + v_block_size + x_block_size

    # Build sparse structure manually for speed
    rows = []
    cols = []
    data = []

    # Helper to append one block
    def append_block(row_offset, col_offset, block, factor=1.0):
        r, c = block.shape
        rr = np.arange(r) + row_offset
        cc = np.arange(c) + col_offset
        for i_idx, i in enumerate(rr):
            for j_idx, j in enumerate(cc):
                rows.append(i)
                cols.append(j)
                data.append(factor * block[i_idx, j_idx])

    # Dynamics equations
    row_offset = 0
    # left side: x[t+1] - A @ x[t] - B @ w[t] = 0
    for t in range(N):
        # variable indices
        w_col = t * p
        x_col_curr = N * n + t * n  # x[t]
        x_col_next = N * n + (t + 1) * n
        # coefficient for x_next: 1
        if t < N:
            # X_next coefficient
            rows.extend([row_offset + i for i in range(n)])
            cols.extend([x_col_next + i for i in range(n)])
            data.extend([1.0] * n)
        # coefficient for -A @ x[t]
        rows.extend([row_offset + i for i in range(n)])
        cols.extend([x_col_curr + i for i in range(n)])
        data.extend([-A[i, j] for i in range(n) for j in range(n)])
        # coefficient for -B @ w[t]
        rows.extend([row_offset + i for i in range(n)])
        cols.extend([w_col + i for i in range(p)])
        data.extend([-B[i, j] for i in range(n) for j in range(p)])
        row_offset += n

    # Measurement equations: y[t] - C @ x[t] - v[t] = 0
    for t in range(N):
        # variable indices
        v_col = w_block_size + t * m
        x_col = N * n + t * n
        # coefficient for -C @ x[t]
        rows.extend([row_offset + i for i in range(m)])
        cols.extend([x_col + i for i in range(n)])
        data.extend([-C[i, j] for i in range(m) for j in range(n)])
        # coefficient for -I @ v[t]
        rows.extend([row_offset + i for i in range(m)])
        cols.extend([v_col + i for i in range(m)])
        data.extend([-1.0] * m)
        # RHS: y[t]
        # will build RHS later
        row_offset += m

    # RHS vector
    rhs = np.zeros(row_offset)
    row_offset = 0
    for t in range(N):
        rhs[row_offset:row_offset + n] = -A @ (x0 if t == 0 else None)  # ignore, will overwrite
        row_offset += n
    # Actually easier to rebuild using concatenated approach
    # Instead, use scipy.sparse for speed but here use dense due to moderate size

    # Build full matrix Aeq and rhs
    # Let's do direct approach again simpler:

    # x next depends on previous x and w, we can eliminate x sequentially
    # compute x[t] explicitly as function of x0 and w[0..t-1]
    Xs = [x0]
    for t in range(N):
        Xs.append(A @ Xs[-1] + B @ np.zeros(p))  # placeholder w
    # Now express y constraints in terms of w only
    # We'll build matrix for w and v

    # Build matrix M such that M * [w; v] = g
    M = np.zeros((N * m, N * p + N * m))
    d = np.zeros(N * m)

    for t in range(N):
        # y[t] = C @ x[t] + v[t]
        # x[t] depends on w[0..t-1]
        # compute coefficient of w[k]
        for k in range(t):
            coef = C @ np.linalg.matrix_power(A, t - k - 1) @ B
            M[t*m:(t+1)*m, k*p:(k+1)*p] += coef
        # v[t] coefficient
        M[t*m:(t+1)*m, w_block_size + t*m:t*m + m] += np.eye(m)
        d[t*m:(t+1)*m] = y[t]

    # right hand side also has x0 term
    d -= C @ Xs[0]
    for t in range(1, N):
        d[t*m:(t+1)*m] -= C @ Xs[t]  # but Xs[t] already includes B*w, so ignore

    # Objective: 0.5*||w||^2 + 0.5*tau*||v||^2
    # Solve normal equations with regularization
    H = np.zeros((K, K))
    H[:w_block_size, :w_block_size] = np.eye(w_block_size)
    H[w_block_size:w_block_size+v_block_size, w_block_size:w_block_size+v_block_size] = tau * np.eye(v_block_size)
    # ignore x part as free due to equality constraints

    # Augmented system for least squares: [H  M^T; M  0] [x; λ] = [0; d]
    # Solve via numpy.linalg.lstsq
    A_big = np.block([[H, M.T], [M, np.zeros((N*m, N*m))]])
    b_big = np.concatenate([np.zeros(K), d])
    sol, *_ = np.linalg.lstsq(A_big, b_big, rcond=None)
    w_sol = sol[:w_block_size].reshape((N, p))
    v_sol = sol[w_block_size:w_block_size+v_block_size].reshape((N, m))
    # compute x trajectory
    x_vals = [x0]
    for t in range(N):
        x_vals.append(A @ x_vals[-1] + B @ w_sol[t])

    return {
        "x_hat": np.array(x_vals[1:]).tolist(),
        "w_hat": w_sol.tolist(),
        "v_hat": v_sol.tolist(),
    }