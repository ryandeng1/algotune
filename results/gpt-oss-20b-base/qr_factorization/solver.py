# solver.py

import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, Dict[str, Any]]:
        """
        Compute the QR factorization of an (n x (n+1)) matrix A.
        Uses numpy.linalg.qr with mode='reduced' for efficient computation.

        Parameters
        ----------
        problem : dict[str, np.ndarray]
            Dictionary containing the input matrix under the key 'matrix'.

        Returns
        -------
        dict
            Dictionary with key 'QR' mapping to {'Q': Q.tolist(), 'R': R.tolist()}.
        """
        A = problem["matrix"]
        # Compute reduced QR factorization (Q is n x n, R is n x (n+1))
        Q, R = np.linalg.qr(A, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}
