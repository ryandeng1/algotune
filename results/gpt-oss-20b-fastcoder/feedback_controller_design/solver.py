from typing import Any
import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Computes a stabilizing feedback gain K and the Lyapunov matrix P for the
        continuous‑time pair (A,B) using the LQR solution with Q=I, R=I.
        """
        A = np.atleast_2d(np.asarray(problem['A'], dtype=float))
        B = np.atleast_2d(np.asarray(problem['B'], dtype=float))
        n = A.shape[0]
        m = B.shape[1]

        # Check controllability rank
        ctrl_mat = B
        for i in range(1, n):
            ctrl_mat = np.concatenate((ctrl_mat, np.linalg.matrix_power(A, i) @ B), axis=1)
        if np.linalg.matrix_rank(ctrl_mat) < n:
            return {'is_stabilizable': False, 'K': None, 'P': None}

        # Solve Continuous Algebraic Riccati Equation with Q=I, R=I
        try:
            P = la.solve_continuous_are(A, B, np.eye(n), np.eye(m))
        except la.LinAlgError:
            return {'is_stabilizable': False, 'K': None, 'P': None}

        # LQR optimal state feedback gain
        K = np.linalg.inv(np.eye(m)) @ B.T @ P

        return {
            'is_stabilizable': True,
            'K': K.astype(float).tolist(),
            'P': P.astype(float).tolist()
        }