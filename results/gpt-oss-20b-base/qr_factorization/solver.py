import numpy as np
from typing import Dict, List, Any

class Solver:
    """
    QR factorization solver optimized for speed.

    The implementation takes advantage of NumPy's highly‑optimized
    LAPACK backend, which is already written in C and Fortran.  No
    additional JIT compilation or Python‑level loops are required.
    """
    def __call__(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List[List[float]]]]:
        """
        Compute the reduced QR factorization of the input matrix `A`

        Parameters
        ----------
        problem : dict
            Dictionary with a single key ``'matrix'`` mapping to a
            2‑D np.ndarray `A` of shape (m, n) with ``m >= n``.

        Returns
        -------
        dict
            Mapping ``'QR'`` -> dict containing lists of lists:
            * ``'Q'`` : The orthonormal matrix Q.
            * ``'R'`` : The upper‑triangular matrix R.

        Notes
        -----
        The result is converted to plain Python lists to ensure that
        the solution can be serialised (e.g., as JSON) without
        depending on Numpy.  The heavy lifting is done entirely in
        compiled C.
        """
        A = problem['matrix']
        # NumPy uses a low‑level LAPACK routine internally which is very fast.
        Q, R = np.linalg.qr(A, mode="reduced")
        return {"QR": {"Q": Q.tolist(), "R": R.tolist()}}