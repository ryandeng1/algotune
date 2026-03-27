import numpy as np
from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List[List[float]]]]:
        """
        Compute the Cholesky factorisation of the given matrix A:
        A = L Lᵗ

        Parameters
        ----------
        problem : dict
            Dictionary containing the key "matrix" whose value is a square, symmetric,
            positive‑definite ``numpy.ndarray`` of type float.

        Returns
        -------
        dict
            `{ "Cholesky": { "L": L.tolist() } }`, where `L` is the lower‑triangular
            factor returned by :func:`numpy.linalg.cholesky`.

        Notes
        -----
        * The conversion to list is performed once, immediately after the
          decomposition, to keep the numerical kernel (NumPy) lightweight.
        * If the matrix is not positive‑definite, a ``numpy.linalg.LinAlgError`` is
          propagated to keep the behaviour identical to the original implementation.
        """
        A = problem["matrix"]
        L = np.linalg.cholesky(A)
        return {"Cholesky": {"L": L.tolist()}}