import numpy as np
import cvxpy as cp
from typing import Any


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the feedback controller design problem using semidefinite programming.

        Args:
            problem: A dictionary containing the system matrices A and B.

        Returns:
            A dictionary containing:
                - K: The feedback gain matrix
                - P: The Lyapunov matrix
        """
        A: np.ndarray = np.asarray(problem["A"])
        B: np.ndarray = np.asarray(problem["B"])
        n, m = A.shape[0], B.shape[1]

        # Decision variables
        Q = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))

        # Build the left‑hand side of the block matrix once as a python expression
        AB = A @ Q + B @ L
        AQ_T = Q @ A.T + L.T @ B.T
        top_row = cp.hstack([Q, AQ_T])
        bot_row = cp.hstack([AB, Q])
        blk = cp.vstack([top_row, bot_row])

        # Constraints
        constraints = [
            blk >> np.eye(2 * n),   # LMI
            Q >> np.eye(n),         # Strictly positive definite
        ]

        # Solve (use a fast, robust solver; SCS is usually available)
        prob = cp.Problem(cp.Minimize(0), constraints)
        try:
            prob.solve(solver=cp.SCS, eps=1e-10, verbose=False, max_iters=5000)
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return {"is_stabilizable": False, "K": None, "P": None}

        Q_val = Q.value
        L_val = L.value

        # Compute K = L * Q^{-1} more stably with solve
        try:
            K_val = L_val @ np.linalg.solve(Q_val, np.eye(n))
            P_val = np.linalg.solve(Q_val, np.eye(n))
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}

        return {
            "is_stabilizable": True,
            "K": K_val.tolist(),
            "P": P_val.tolist(),
        }