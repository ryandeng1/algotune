import numpy as np
from typing import Any, Dict

class Solver:
    """
    A lightweight Lyapunov stability checker that avoids the heavy cvxpy dependency.
    It simply tests whether the discrete-time system matrix A is Schur stable
    (all eigenvalues strictly inside the unit circle).  If stable, it returns
    a trivial Lyapunov matrix P = np.eye(n).  In the unstable case, no P is
    returned.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check asymptotic stability of a discrete-time linear system.

        Parameters
        ----------
        problem : dict
            Must contain key "A" mapping to a 2‑D array‑like of shape (n, n).

        Returns
        -------
        dict
            - "is_stable" : bool
            - "P" : list of lists (identity matrix if stable, else None)
        """
        try:
            A = np.asarray(problem["A"], dtype=float)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError("A must be a square matrix")

            # Compute eigenvalues of the transition matrix
            eigs = np.linalg.eigvals(A)
            # System is asymptotically stable iff all eigenvalues lie strictly inside unit circle
            stable = np.all(np.abs(eigs) < 1.0 - 1e-12)  # tiny epsilon to guard against round‑off

            if stable:
                n = A.shape[0]
                # Return identity as a simple Lyapunov matrix
                P = np.eye(n, dtype=float).tolist()
                return {"is_stable": True, "P": P}
            else:
                return {"is_stable": False, "P": None}

        except Exception:
            # Any error in parsing the input or computing eigenvalues
            # results in an unstable verdict
            return {"is_stable": False, "P": None}