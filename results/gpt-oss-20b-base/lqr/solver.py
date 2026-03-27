import numpy as np
from scipy.linalg import solve as linalg_solve


def solve(problem: dict[str, any]) -> dict[str, any]:
    """
    Compute optimal control sequence via backward Riccati recursion.

    Returns a dictionary with key "U" (shape (T, m)).
    """
    A, B = problem["A"], problem["B"]
    Q, R, P = problem["Q"], problem["R"], problem["P"]
    T, x0 = problem["T"], problem["x0"]

    n, m = B.shape
    # Pre‑allocate arrays
    S = np.empty((T + 1, n, n), dtype=A.dtype)
    K = np.empty((T, m, n), dtype=B.dtype)
    S[T] = P

    # Backward pass
    for t in range(T - 1, -1, -1):
        St1 = S[t + 1]
        # Solve (R + Bᵀ Stₜ₊₁ B) Kᵀ = Bᵀ Stₜ₊₁ A
        M1 = R + B.T @ St1 @ B
        M2 = B.T @ St1 @ A
        # M1 is symmetric positive‑definite; use direct solver
        L = np.linalg.cholesky(M1)
        y = np.linalg.solve(L, M2)
        K[t] = np.linalg.solve(L.T, y)

        Acl = A - B @ K[t]
        # Riccati update with symmetry enforcement
        S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
        # Make symmetric explicitly
        S[t] = 0.5 * (S[t] + S[t].T)

    # Forward pass
    U = np.empty((T, m), dtype=A.dtype)
    x = x0
    for t in range(T):
        u = -K[t] @ x
        U[t] = u.ravel()
        x = A @ x + B @ u

    return {"U": U}