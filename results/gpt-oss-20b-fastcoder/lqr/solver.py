from typing import Any
import numpy as np
from scipy.linalg import solve as linalg_solve

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute optimal control sequence via backward Riccati recursion.

        Returns dict with key "U" (shape (T, m)).
        """
        A, B, Q, R, P, T, x0 = (
            problem["A"], problem["B"], problem["Q"],
            problem["R"], problem["P"], problem["T"], problem["x0"]
        )
        n, m = B.shape
        S = np.empty((T + 1, n, n))
        K = np.empty((T, m, n))
        S[T] = P.copy()

        # Backward Riccati recursion
        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            M1 = R + B.T @ St1 @ B  # (m,m)
            M2 = B.T @ St1 @ A      # (m,n)
            try:
                K[t] = linalg_solve(M1, M2, assume_a="pos")
            except np.linalg.LinAlgError:
                K[t] = np.linalg.pinv(M1) @ M2
            Acl = A - B @ K[t]
            S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            # ensure symmetry
            S[t] = (S[t] + S[t].T) * 0.5

        # Forward simulation
        U = np.zeros((T, m))
        x = x0.copy()
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.ravel()
            x = A @ x + B @ u

        return {"U": U}