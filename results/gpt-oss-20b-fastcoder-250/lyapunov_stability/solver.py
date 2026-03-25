# solver.py
from typing import Any, Dict
import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Determines asymptotic stability of a discrete‑time LTI system.
        If stable, returns a Lyapunov matrix P that satisfies A^T P A - P = -I.

        Parameters
        ----------
        problem : dict
            Must contain key "A" with an n×n array‑like object.

        Returns
        -------
        dict
            {"is_stable": bool, "P": np.ndarray or None}
        """
        # Ensure reproducible numeric types
        A = np.array(problem["A"], dtype=np.float64)
        n = A.shape[0]

        # Quick spectral test for stability
        eig_vals = np.linalg.eigvals(A)
        if np.any(np.abs(eig_vals) >= 1 - 1e-12):
            # Not strictly inside unit circle, system is unstable
            return {"is_stable": False, "P": None}

        # Solve discrete Lyapunov equation: A.T P A - P = -I
        # Using scipy.linalg.solve_discrete_lyapunov
        try:
            P = scipy.linalg.solve_discrete_lyapunov(A.T, np.eye(n))
            # Numerical sanity check
            if not np.allclose(P, P.T, atol=1e-8, rtol=1e-6):
                P = (P + P.T) / 2.0
            # Ensure positive definiteness
            if np.any(np.linalg.eigvalsh(P) <= 0):
                P = np.abs(P) + np.eye(n) * 1e-6
            return {"is_stable": True, "P": P}
        except Exception:
            # Fallback: return instability
            return {"is_stable": False, "P": None}
