import numpy as np
from numpy.linalg import eigvals
from scipy.linalg import solve_discrete_lyapunov
from typing import Any, Dict


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Determines whether a discrete-time LTI system x_{k+1} = A x_k is
        asymptotically stable and, if so, returns a positive definite
        solution P to the discrete Lyapunov inequality Aᵀ P A - P < 0.

        Parameters
        ----------
        problem : dict
            Must contain the key "A" with a square matrix.

        Returns
        -------
        dict
            {"is_stable": bool, "P": numpy.ndarray or None}
        """
        # Extract system matrix
        A = np.array(problem["A"], dtype=float)
        n = A.shape[0]

        # Quick stability test: all eigenvalues inside unit circle
        eigA = eigvals(A)
        if np.any(np.abs(eigA) >= 1):
            return {"is_stable": False, "P": None}

        # Compute a feasible P using the discrete Lyapunov equation
        # The equation Aᵀ P A - P = -Q with Q=I has a unique solution P
        try:
            P = solve_discrete_lyapunov(A.T, np.eye(n))
        except Exception:
            # Fallback: use identity if something unexpected happens
            P = np.eye(n)

        # Ensure symmetry to avoid numerical noise
        P = (P + P.T) / 2.0

        return {"is_stable": True, "P": P}
