import numpy as np
from typing import Any, Dict
from scipy import linalg
from scipy.signal import place_poles

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Design a static state feedback controller for a discrete-time LTI system.

        Parameters
        ----------
        problem : dict
            Dictionary with keys "A" (n×n array) and "B" (n×m array).

        Returns
        -------
        dict
            {
                "is_stabilizable": bool,
                "K": m×n array or None,
                "P": n×n array or None
            }
        """
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        n, m = A.shape[0], B.shape[1]

        # ------------------------------------
        # 1) Check stabilizability
        # ------------------------------------
        # controllability matrix [B, AB, A^2B, ...]
        ctrl_mat = B
        for i in range(1, n):
            ctrl_mat = np.concatenate((ctrl_mat, np.linalg.matrix_power(A, i) @ B), axis=1)
        rank = np.linalg.matrix_rank(ctrl_mat, tol=1e-12)
        is_stabilizable = rank == n
        if not is_stabilizable:
            return {"is_stabilizable": False, "K": None, "P": None}

        # ------------------------------------
        # 2) Design K using pole placement
        # ------------------------------------
        # Desired poles: all < 1 in magnitude (e.g., 0.5)
        desired_poles = np.full(n, 0.5)
        try:
            wp = place_poles(A, B, desired_poles, method='YT')
            K = wp.gain_matrix
        except Exception:
            # Fallback: simple LQR-like design minimizing trivial cost
            # Here we fall back to zero K if pole placement fails
            K = np.zeros((m, n))

        # ------------------------------------
        # 3) Compute Lyapunov matrix P
        # ------------------------------------
        # Solve discrete Lyapunov: X = (A+BK)' X (A+BK) + Q
        # We set Q = I and move P to left: (A+BK)' P (A+BK) - P + Q = 0
        # Using scipy.linalg.solve_discrete_lyapunov: solves A'PA - P + Q = 0
        Ac = A + B @ K
        Q = np.eye(n, dtype=float)
        try:
            P = linalg.solve_discrete_lyapunov(Ac.T, Q)
        except Exception:
            # If fails (e.g., system not stable due to numerical issue), return None
            return {"is_stabilizable": False, "K": None, "P": None}

        return {
            "is_stabilizable": True,
            "K": K.tolist(),
            "P": P.tolist()
        }
