import numpy as np
from typing import Any, Dict

def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    # Convert inputs to numpy arrays
    A = np.array(problem['A'])
    B = np.array(problem['B'])
    C = np.array(problem['C'])
    y = np.array(problem['y'])
    x0 = np.array(problem['x_initial'])
    tau = float(problem['tau'])

    N, m = y.shape
    n = A.shape[1]
    p = B.shape[1]

    # Pre‑compute powers of A and terms of the dynamics
    A_pows = [np.eye(n, dtype=A.dtype)]
    for _ in range(1, N + 1):
        A_pows.append(A @ A_pows[-1])

    # Build M and d such that v = d - M w
    # M has shape (N*m, N*p), d has shape (N*m,)
    M = np.zeros((N * m, N * p), dtype=A.dtype)
    d = np.zeros(N * m, dtype=A.dtype)

    for t in range(N):
        # index for d (row block)
        idx_d = t * m
        # term C * A^t * x0
        d[idx_d:idx_d + m] = y[t] - C @ A_pows[t] @ x0
        for k in range(t):
            # block for w_k
            idx_wk = k * p
            idx_row = t * m
            # contribution of w_k: C * A^{t-1-k} * B
            M[idx_row:idx_row + m, idx_wk:idx_wk + p] = C @ A_pows[t - 1 - k] @ B

    # Solve (I + tau M^T M) w = tau M^T d
    I = np.eye(N * p, dtype=A.dtype)
    lhs = I + tau * (M.T @ M)
    rhs = tau * (M.T @ d)
    try:
        w_vec = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return {'x_hat': [], 'w_hat': [], 'v_hat': []}

    w = w_vec.reshape(N, p)
    v = d - M @ w_vec
    v = v.reshape(N, m)

    # Recover state trajectory
    x = np.empty((N + 1, n), dtype=A.dtype)
    x[0] = x0
    for t in range(N):
        x[t + 1] = A @ x[t] + B @ w[t]

    return {
        'x_hat': x.tolist(),
        'w_hat': w.tolist(),
        'v_hat': v.tolist()
    }