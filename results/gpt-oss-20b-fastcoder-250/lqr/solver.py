# solver.py
import numpy as np
from typing import Any
from scipy.linalg import solve as linalg_solve

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the optimal control sequence for a discrete-time LQR problem.
        This implementation uses the back‑propagation Riccati recursion
        and forwards simulation.  All arithmetic is performed with
        double precision floating point for numerical stability.
        """
        # Extract problem data
        A: np.ndarray = np.asarray(problem["A"], dtype=np.float64)
        B: np.ndarray = np.asarray(problem["B"], dtype=np.float64)
        Q: np.ndarray = np.asarray(problem["Q"], dtype=np.float64)
        R: np.ndarray = np.asarray(problem["R"], dtype=np.float64)
        P: np.ndarray = np.asarray(problem["P"], dtype=np.float64)
        T: int = int(problem["T"])
        x0: np.ndarray = np.asarray(problem["x0"], dtype=np.float64).reshape(-1, 1)

        n, m = B.shape

        # Precompute transposes for speed
        Bt = B.T
        Qt = Q.T  # not needed but keep symmetry

        # Riccati recursion
        S = P.copy()
        K = np.empty((T, m, n), dtype=np.float64)
        for t in range(T - 1, -1, -1):
            # M1 = R + B.T @ S @ B
            M1 = R + Bt @ S @ B
            # M2 = B.T @ S @ A
            M2 = Bt @ S @ A
            try:
                Kt = linalg_solve(M1, M2, assume_a="pos")
            except np.linalg.LinAlgError:
                Kt = np.linalg.pinv(M1) @ M2
            K[t] = Kt
            # Closed‑loop A matrix
            Acl = A - B @ Kt
            # Update Riccati matrix
            S = Q + Kt.T @ R @ Kt + Acl.T @ S @ Acl
            # Ensure symmetry
            S = (S + S.T) * 0.5

        # Forward simulation to obtain control sequence
        U = np.empty((T, m), dtype=np.float64)
        x = x0
        for t in range(T):
            u = -K[t] @ x
            U[t] = u.ravel()
            x = A @ x + B @ u

        return {"U": U}
