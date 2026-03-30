# solver.py
import numpy as np
from typing import Any
from scipy.linalg import solve as linalg_solve
import numba

# A small helper that uses Cholesky for positive definite matrices.
@numba.njit
def _cholesky_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A x = B for x using Cholesky decomposition (assuming A is SPD)."""
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, B)
    x = np.linalg.solve(L.T, y)
    return x


class Solver:

    @staticmethod
    @numba.njit
    def _backward_riccati(A: np.ndarray,
                          B: np.ndarray,
                          Q: np.ndarray,
                          R: np.ndarray,
                          P: np.ndarray,
                          T: int):
        """Perform the back‑propagation part of the LQR in compiled code."""
        n, m = B.shape
        S = np.zeros((T + 1, n, n))
        K = np.zeros((T, m, n))

        S[T] = P

        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            # Compute M1 = R + B^T S[t+1] B and M2 = B^T S[t+1] A
            M1 = R + B.T @ St1 @ B
            M2 = B.T @ St1 @ A

            # Solve for K[t] using Cholesky (robust for SPD M1)
            K[t] = _cholesky_solve(M1, M2)

            # Closed‑loop dynamics and Riccati step
            Acl = A - B @ K[t]
            S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            # Keep symmetry (minor numerical disturbance)
            S[t] = (S[t] + S[t].T) * 0.5

        return K

    @staticmethod
    @numba.njit
    def _forward_control(A: np.ndarray,
                         B: np.ndarray,
                         K: np.ndarray,
                         x0: np.ndarray,
                         T: int):
        """Given K compute control sequence and state trajectory."""
        m, n = B.shape[1], A.shape[0]
        U = np.zeros((T, m))
        x = x0.copy()
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.reshape(-1)
            x = A @ x + B @ u
        return U

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Compute optimal control sequence via Riccati recursion."""
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]
        R = problem["R"]
        P = problem["P"]
        T = problem["T"]
        x0 = problem["x0"]

        # Backward sweep
        K = self._backward_riccati(A, B, Q, R, P, T)

        # Forward simulation
        U = self._forward_control(A, B, K, x0, T)

        return {"U": U}