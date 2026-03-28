from typing import Any, Dict
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eigvals

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines asymptotic stability of a linear system described by matrix A.
        If the system is stable, returns a Lyapunov matrix P that satisfies
        A^T P + P A = -I.  The routine uses a direct spectral check and
        the continuous‐time Lyapunov solver from SciPy (which is much faster
        than a generic SDP solver).

        Args:
            problem: A dictionary containing the system matrix `A`.

        Returns:
            dict with keys:
                - 'is_stable': bool
                - 'P': list representation of the Lyapunov matrix (None if unstable)
        """
        A = np.asarray(problem["A"], dtype=float)
        # Quick spectral test: all eigenvalues must have strictly negative real part
        if np.any(np.real(eigvals(A)) >= 0):
            return {"is_stable": False, "P": None}

        # Solve the Lyapunov equation A^T P + P A = -I
        try:
            P = solve_continuous_lyapunov(A.T, -np.eye(A.shape[0]))
            # Ensure symmetry (numerical errors can make it slightly asymmetric)
            P = (P + P.T) / 2.0
        except Exception:
            return {"is_stable": False, "P": None}

        return {"is_stable": True, "P": P.tolist()}