import numpy as np
from typing import Any, Dict

class Solver:
    """
    Fast QR factorization solver.
    Uses NumPy's LAPACK-backed qr implementation which is already highly optimised.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Compute the reduced QR factorization A = Q @ R for the matrix supplied in *problem*.

        Parameters
        ----------
        problem : dict
            Must contain a key ``"matrix"`` mapping to a 2‑D NumPy array.

        Returns
        -------
        dict
            ``{"QR": {"Q": Q_list, "R": R_list}}`` where the lists are regular Python
            nested lists containing the elements of Q and R.
        """
        A = problem["matrix"]

        # LAPACK-backed QR, mode='reduced' returns Q with orthonormal columns
        Q, R = np.linalg.qr(A, mode="reduced")

        # Convert to plain Python lists for serialisation / output consistency.
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}