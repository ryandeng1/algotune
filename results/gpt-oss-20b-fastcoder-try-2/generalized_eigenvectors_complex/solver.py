import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from typing import Tuple, List

class Solver:
    def solve(self, problem: Tuple[NDArray[np.float64], NDArray[np.float64]]
             ) -> Tuple[List[complex], List[List[complex]]]:
        """
        Solve the generalized eigenvalue problem ``A x = λ B x``.

        The function scales ``B`` to improve numerical stability,
        solves the scaled problem, normalises the eigenvectors,
        sorts the results by decreasing real part and then
        imaginary part of the eigenvalues, and returns
        the eigenvalues and corresponding unit‑norm eigenvectors
        as Python lists.
        """
        A, B = problem
        # --- scaling ------------------------------------------------------
        scale = np.sqrt(np.linalg.norm(B, ord="fro"))
        A_s = A / scale
        B_s = B / scale

        # --- solve --------------------------------------------------------
        w, v = la.eig(A_s, B_s)

        # --- normalise eigenvectors ---------------------------------------
        norms = np.linalg.norm(v, axis=0)
        # avoid division by zero
        good = norms > 1e-15
        v[:, good] /= norms[good]  # broadcast divisor

        # --- sort ---------------------------------------------------------
        idx = np.argsort(
            np.lexsort((-w.imag, -w.real))  # descending
        )
        w = list(w[idx])
        v = v[:, idx].T  # transpose to list rows

        # --- convert to Python lists --------------------------------------
        eigenvalues = [complex(val) for val in w]
        eigenvectors = [list(vec) for vec in v]

        return eigenvalues, eigenvectors