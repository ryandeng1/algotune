import numpy as np
from scipy.linalg import solve_continuous_are, inv

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fast LQR-based stabilizing controller design using the continuous‑time
        Algebraic Riccati Equation (ARE).  This replaces the SDP formulation
        with a direct analytic solution that is orders of magnitude faster.

        Parameters
        ----------
        problem : dict
            Must contain:
                * 'A' : ndarray (n x n)
                * 'B' : ndarray (n x m)

        Returns
        -------
        result : dict
            Keys:
                * 'is_stabilizable' : bool
                * 'K'                : list[list[float]] or None
                * 'P'                : list[list[float]] or None
        """
        try:
            A = np.asarray(problem['A'], dtype=float)
            B = np.asarray(problem['B'], dtype=float)
            n, m = A.shape[0], B.shape[1]

            # Use Q = I, R = I for a simple stabilizing controller
            Q = np.eye(n)
            R = np.eye(m)

            # Solve the continuous‑time ARE
            P = solve_continuous_are(A, B, Q, R)

            # Compute the state‑feedback gain K
            K = np.linalg.solve(R, B.T @ P)

            # Return results as plain Python lists
            return {
                'is_stabilizable': True,
                'K': K.tolist(),
                'P': P.tolist(),
            }

        except Exception:
            # If the ARE cannot be solved, declare system not stabilizable
            return {
                'is_stabilizable': False,
                'K': None,
                'P': None,
            }