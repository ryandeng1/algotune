# solver.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np
from scipy.linalg import solve as linalg_solve


class Solver:

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes the optimal control sequence U for a discrete‐time LQR problem
        using a fast backward Riccati recursion.
        """
        # Unpack problem data
        A: np.ndarray = problem["A"]
        B: np.ndarray = problem["B"]
        Q: np.ndarray = problem["Q"]
        R: np.ndarray = problem["R"]
        P: np.ndarray = problem["P"]
        T: int = problem["T"]
        x0: np.ndarray = problem["x0"]

        n: int = B.shape[0]
        m: int = B.shape[1]

        # Allocate containers
        S = np.empty((T + 1, n, n))
        K = np.empty((T, m, n))
        S[T] = P.copy()

        # Backward Riccati recursion
        for t in range(T - 1, -1, -1):
            St1 = S[t + 1]

            M1 = R + B.T @ St1 @ B                # (m, m)
            M2 = B.T @ St1 @ A                    # (m, n)

            try:
                K[t] = linalg_solve(M1, M2, assume_a="pos")
            except np.linalg.LinAlgError:
                # Fall back to pseudo‑inverse if M1 is singular or not
                # positive definite.  This branch is rare in well‑posed
                # control problems.
                K[t] = np.linalg.pinv(M1) @ M2

            # Closed‑loop dynamics and update S
            Acl = A - B @ K[t]
            St = Q + K[t].T @ R @ K[t] + Acl.T @ St1 @ Acl
            S[t] = (St + St.T) * 0.5  # enforce symmetry

        # Forward simulation to compute control sequence
        U = np.empty((T, m))
        x = x0.copy()
        for t in range(T):
            u = -K[t] @ x          # (m,)
            U[t] = u
            x = A @ x + B @ u

        return {"U": U}