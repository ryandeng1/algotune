import numpy as np
from scipy.linalg import cholesky, solve_triangular

def solve(problem: dict):
    """Compute optimal control sequence via backward Riccati recursion."""
    A, B, Q, R, P, T, x0 = (
        problem["A"],
        problem["B"],
        problem["Q"],
        problem["R"],
        problem["P"],
        problem["T"],
        problem["x0"],
    )
    n, m = B.shape

    # Allocate arrays
    S = np.empty((T + 1, n, n), dtype=A.dtype)
    K = np.empty((T, m, n), dtype=A.dtype)
    S[T] = P

    # Backward Riccati recursion
    for t in range(T - 1, -1, -1):
        St1 = S[t + 1]
        # M1 = R + B^T * St1 * B
        M1 = R + B.T @ St1 @ B
        # Cholesky for efficient solve; skip if fails
        L = None
        try:
            L = cholesky(M1, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            # Fallback to direct solve
            K[t] = np.linalg.solve(M1, B.T @ St1 @ A)
        else:
            # Solve L * y = B^T * St1 * A
            Y = solve_triangular(L, B.T @ St1 @ A, lower=True, check_finite=False)
            # Solve L^T * K = Y
            K[t] = solve_triangular(L.T, Y, lower=False, check_finite=False)

        # Closed-loop A
        Acl = A - B @ K[t]
        # Updated S
        # S[t] = Q + K^T R K + Acl^T St1 Acl
        St = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
        # Symmetrize
        S[t] = (St + St.T) * 0.5

    # Forward simulation to compute U
    U = np.empty((T, m), dtype=A.dtype)
    x = x0
    for t in range(T):
        u = -K[t] @ x
        U[t] = u.ravel()
        x = A @ x + B @ u

    return {"U": U}