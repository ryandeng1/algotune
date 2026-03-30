from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

class Solver:
    """
    Compute a truncated SVD of the input matrix.

    This implementation uses NumPy's fast LAPACK-backed SVD routine instead of
    Scikit‑Learn's `randomized_svd`, which is considerably slower for dense
    matrices.  The routine returns the first `n_components` singular vectors
    and values, matching the requested number of components.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Parameters
        ----------
        problem : dict
            Must contain keys:
            - 'matrix': 2‑D array or array‑like convertible to a NumPy array.
            - 'n_components': int, number of singular components to retain.
            - 'matrix_type': str, ignored but kept for API compatibility.

        Returns
        -------
        dict
            Keys 'U', 'S', 'V' hold the truncated SVD components.
        """
        A = np.asarray(problem["matrix"])
        n_components = int(problem["n_components"])

        # Compute the full SVD via LAPACK (fast for dense matrices).
        # `full_matrices=False` gives right dimensions for the truncated result.
        U, s, Vh = np.linalg.svd(A, full_matrices=False)

        # Truncate to requested number of components.
        U_trunc = U[:, :n_components]
        s_trunc = s[:n_components]
        V_trunc = Vh[:n_components, :].T  # transpose to match original spec

        return {"U": U_trunc, "S": s_trunc, "V": V_trunc}