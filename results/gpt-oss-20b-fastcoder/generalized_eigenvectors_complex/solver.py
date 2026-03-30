# solver.py
from __future__ import annotations

from typing import Tuple, List, Any
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: Tuple[NDArray[np.float64], NDArray[np.float64]]
              ) -> Tuple[List[complex], List[List[complex]]]:
        """
        Solve A·x = λ B·x for real square matrices A,B.
        The eigenvalues are returned sorted by descending real part,
        then by descending imaginary part.  Eigenvectors are unit‑norm
        and returned as nested Python lists.
        """
        A, B = problem

        # --- Numerical scaling ------------------------------------
        # Use the Frobenius norm of B for a stable scaling factor.
        scale_B = np.sqrt(np.linalg.norm(B, ord='fro'))
        A_scaled = A / scale_B
        B_scaled = B / scale_B

        # --- Eigen‑decomposition -----------------------------------
        e_vals, e_vecs = la.eig(A_scaled, B_scaled)

        # --- Normalise eigenvectors --------------------------------
        norms = la.norm(e_vecs, axis=0)
        # protect against zero‐length columns
        zero_norms = norms < 1e-15
        if zero_norms.any():
            # leave these columns unchanged, they are already zero vectors
            norms[zero_norms] = 1.0
        e_vecs = e_vecs / norms

        # --- Sort by decreasing real part, then imaginary part ----
        # Create a mixed key for sorting.
        sort_keys = np.column_stack(
            (-e_vals.real, -e_vals.imag)
        ).astype(np.float64)
        order = np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))
        e_vals_sorted = e_vals[order]
        e_vecs_sorted = e_vecs[:, order]

        # --- Convert to Python lists for the required return format
        eigenvalues_list = e_vals_sorted.tolist()
        eigenvectors_list = [vec.tolist() for vec in e_vecs_sorted.T]

        return (eigenvalues_list, eigenvectors_list)