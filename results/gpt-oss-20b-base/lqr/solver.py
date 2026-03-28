import numpy as np
from scipy.linalg import cho_factor, cho_solve

def solve(problem: dict) -> dict:
    """
    Backward Riccati recursion to compute optimal control sequence.
    Returns dictionary with key `"U"` (shape (T, m)).
    """
    A = problem['A']          # (n, n)
    B = problem['B']          # (n, m)
    Q = problem['Q']          # (n, n)
    R = problem['R']          # (m, m)
    P = problem['P']          # (n, n)
    T = problem['T']          # int
    x0 = problem['x0']        # (n,)

    n, m = B.shape
    # Riccati matrices
    S = np.empty((T + 1, n, n), dtype=A.dtype)
    K = np.empty((T, m, n), dtype=A.dtype)
    S[T] = P

    for t in range(T - 1, -1, -1):
        St1 = S[t + 1]
        # M1 = R + B.T @ St1 @ B
        BTSt1 = B.T @ St1
        M1 = R + BTSt1 @ B
        # K[t] = M1^{-1} @ (B.T @ St1 @ A)
        rhs = BTSt1 @ A
        # Use Cholesky for speed and numerical stability
        c, lower = cho_factor(M1)
        K[t] = cho_solve((c, lower), rhs)

        # Closed‑loop dynamics
        Acl = A - B @ K[t]
        # Update Riccati matrix
        S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
        # Symmetrise to guard against round‑off
        S[t] = (S[t] + S[t].T) * 0.5

    U = np.empty((T, m), dtype=A.dtype)
    x = x0
    for t in range(T):
        u = -K[t] @ x
        U[t] = u
        x = A @ x + B @ u

    return {'U': U}