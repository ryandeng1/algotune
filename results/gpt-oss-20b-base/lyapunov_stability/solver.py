from __future__ import annotations
from typing import Any, Dict

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, eigvals

class Solver:
    """Timely solver for discrete–time Lyapunov stability.

    The original implementation used CVXPY and a generic SDP solver.  
    For this specific problem we can use analytical formulas:

        1. The system A is asymptotically stable iff all eigenvalues of
           A lie inside the unit disc.

        2. When this holds the Lyapunov matrix is the unique solution of
           P = Aᵀ P A + I which can be found by solving a discrete
           Lyapunov equation with ``scipy.linalg.solve_discrete_lyapunov``.

    This eliminates the overhead of an external LP/SDP solver and makes
    the routine both faster and deterministic.
    """

    @staticmethod
    def _is_asymptotically_stable(A: np.ndarray) -> bool:
        """Return True iff all eigenvalues of A have |λ|<1."""
        return np.all(np.abs(eigvals(A)) < 1)

    @staticmethod
    def _lyapunov_matrix(A: np.ndarray) -> np.ndarray | None:
        """Return the solution to P = Aᵀ P A + I,
        or ``None`` if the Lyapunov equation cannot be solved
        (should never happen for a stable A)."""
        try:
            P = solve_discrete_lyapunov(A.T, np.eye(A.shape[0]))
            # force symmetry due to numerical noise
            return (P + P.T) / 2.0
        except Exception:
            return None

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        A = np.array(problem["A"], dtype=float)
        if not self._is_asymptotically_stable(A):
            return {"is_stable": False, "P": None}

        P = self._lyapunov_matrix(A)
        if P is None:
            return {"is_stable": False, "P": None}

        return {"is_stable": True, "P": P.tolist()}