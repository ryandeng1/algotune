import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve


def solve(problem: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute optimal control sequence via backward Riccati recursion.

    Returns dict with key "U" (shape (T, m)).
    """
    A, B = problem["A"], problem["B"]
    Q, R, P = problem["Q"], problem["R"], problem["P"]
    T, x0 = problem["T"], problem["x0"]

    n, m = B.shape

    # Pre‑allocate array for cost matrix and feedback gain
    S = np.empty((T + 1, n, n))
    K = np.empty((T, m, n))
    S[T] = P
    R_mat = R  # keep reference

    for t in range(T - 1, -1, -1):
        St1 = S[t + 1]
        # Build M1 = R + B.T @ St1 @ B
        M1 = R_mat + B.T @ St1 @ B
        # Build M2 = B.T @ St1 @ A
        M2 = B.T @ St1 @ A

        # Solve M1 * K[t].T = M2   =>   K[t] = (M1^{-1} * M2).T
        # Use Cholesky for positive definite M1
        c, lower = cho_factor(M1, lower=True, check_finite=False)
        K_t_T = cho_solve((c, lower), M2, check_finite=False)
        K[t] = K_t_T.T

        # Closed‑loop dynamics
        Acl = A - B @ K[t]
        # Riccati update (only upper‑triangular part, then symmetrise)
        S_t = Q + K[t].T @ R_mat @ K[t] + Acl.T @ St1 @ Acl
        S[t] = (S_t + S_t.T) * 0.5

    # Forward rollout of optimal control
    U = np.empty((T, m))
    x = x0
    for t in range(T):
        u = -K[t] @ x
        U[t] = u.ravel()
        x = A @ x + B @ u

    return {"U": U}