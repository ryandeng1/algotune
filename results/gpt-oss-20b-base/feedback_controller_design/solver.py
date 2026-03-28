from typing import Any
import numpy as np
from scipy.linalg import solve_continuous_are, inv, lstsq

class Solver:
    """
    A fast feedback controller designer that uses the continuous–time
    algebraic Riccati equation (CARE) to compute a stabilising state‑feedback
    gain K and the associated Lyapunov matrix P.  The algorithm is O(n³)
    and runs in a handful of microseconds on typical problem sizes.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the feedback controller design problem.

        Parameters
        ----------
        problem : dict
            Must contain the system matrices ``A`` and ``B`` as arrays or
            list‑of‑lists.

        Returns
        -------
        dict
            ``is_stabilizable`` : bool – whether a stabilising controller exists
            ``K``                : list of list – state‑feedback gain matrix
            ``P``                : list of list – Lyapunov matrix
        """
        # Parse input matrices
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)

        try:
            # Solve the continuous‑time algebraic Riccati equation
            # with Q = I and R = I (the cheapest command‑signal penalty).
            P = solve_continuous_are(A, B, np.eye(A.shape[0]), np.eye(B.shape[1]))
            # Check if solution is positive definite (i.e., stabilising)
            if not np.all(np.linalg.eigvals(P) > 0):
                raise ValueError("P not positive definite")

            # Compute state‑feedback gain K = Bᵀ P
            # (equivalently R⁻¹ Bᵀ P with R=I)
            K = B.T @ P
        except Exception:
            return {
                "is_stabilizable": False,
                "K": None,
                "P": None,
            }

        return {
            "is_stabilizable": True,
            "K": K.tolist(),
            "P": P.tolist(),
        }