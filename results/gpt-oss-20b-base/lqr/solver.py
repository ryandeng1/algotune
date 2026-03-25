# solver.py
from typing import Any, Dict
import numpy as np
from scipy.linalg import solve as linalg_solve


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Compute optimal LQR control sequence via backward Riccati recursion."""

        # Extract matrices and parameters
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        Q = np.asarray(problem["Q"], dtype=np.float64)
        R = np.asarray(problem["R"], dtype=np.float64)
        P = np.asarray(problem["P"], dtype=np.float64)
        T = int(problem["T"])
        x0 = np.asarray(problem["x0"], dtype=np.float64).reshape(-1, 1)

        n, m = B.shape
        # Prepare arrays for Riccati matrices and feedback gains
        S = np.zeros((T + 1, n, n), dtype=np.float64)
        K = np.zeros((T, m, n), dtype=np.float64)
        S[T] = P

        # Backward Riccati recursion
        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]
            M1 = R + B.T @ St1 @ B
            M2 = B.T @ St1 @ A
            try:
                # Solve M1 * X = M2
                K[t] = linalg_solve(M1, M2, assume_a="pos")
            except np.linalg.LinAlgError:
                K[t] = np.linalg.pinv(M1) @ M2
            Acl = A - B @ K[t]
            S[t] = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            # Symmetrize to avoid numerical drift
            S[t] = (S[t] + S[t].T) * 0.5

        # Forward simulation to build control sequence
        U = np.zeros((T, m), dtype=np.float64)
        x = x0
        for t in range(T):
            u = -K[t] @ x  # shape (m,1)
            U[t] = u.ravel()  # store as 1D vector
            x = A @ x + B @ u

        return {"U": U}
