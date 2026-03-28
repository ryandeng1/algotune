from typing import Any
import numpy as np

# ------------------------------------------------------------------
#  Author: Performance Engineer
#  Approach: hand‑rolled Riccati recursion with Numba JIT
#  The core loop is compiled to machine code, avoiding the overhead of
#  Python loops and the repeated `np.linalg.solve` calls.
# ------------------------------------------------------------------

from numba import njit, prange, float64, int32
from numpy import zeros, eye

@njit
def _riccati_forward(A, B, Q, R, P, T, x0):
    """
    Compute optimal control law U for a finite horizon LQR problem
    using backward Riccati recursion (Numba accelerated).
    """
    n = B.shape[0]
    m = B.shape[1]
    S = zeros((T + 1, n, n), dtype=float64)
    K = zeros((T, m, n), dtype=float64)
    S[T] = P

    for t in range(T - 1, -1, -1):
        St1 = S[t + 1]
        # M1 = R + B^T St1 B
        M1 = R + B.T @ St1 @ B
        # M2 = B^T St1 A
        M2 = B.T @ St1 @ A
        # Solve M1 K^T = M2   ->   K = (M1^-1 M2)^T
        K_t = np.linalg.solve(M1, M2).T
        K[t] = K_t

        # (A - B K)^T  S[t+1]  (A - B K)
        Acl_T = A.T - B.T @ K_t.T
        Acl = A - B @ K_t
        S_t = Q + K_t @ R @ K_t + Acl_T @ St1 @ Acl
        # Enforce symmetry
        S[t] = 0.5 * (S_t + S_t.T)

    U = zeros((T, m), dtype=float64)
    x = x0.copy()
    for t in range(T):
        u = -K[t] @ x
        U[t] = u
        x = A @ x + B @ u
    return U


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = problem['A'].astype(np.float64)
        B = problem['B'].astype(np.float64)
        Q = problem['Q'].astype(np.float64)
        R = problem['R'].astype(np.float64)
        P = problem['P'].astype(np.float64)
        T = problem['T']
        x0 = problem['x0'].astype(np.float64)

        U = _riccati_forward(A, B, Q, R, P, T, x0)
        return {"U": U}