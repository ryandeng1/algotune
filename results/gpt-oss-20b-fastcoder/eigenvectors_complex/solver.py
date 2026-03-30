from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    """
    Fast eigenvector solver.

    The implementation relies purely on NumPy vectorised operations:
    * `np.linalg.eig` gives the eigenvalues and eigenvectors.
    * Sorting is performed once by generating a vectorised sorting index.
    * Normalisation is done in one go for all eigenvectors.
    The result is converted to a Python list of lists only at the very end,
    ensuring that the heavy work stays in compiled NumPy code.
    """

    def solve(self, problem: NDArray) -> list[list[complex]]:
        """
        Solve the eigenvector problem for the given non‑symmetric matrix.
        Compute eigenvalues and eigenvectors using np.linalg.eig.
        Sort the eigenpairs in descending order by the real part (and then
        imaginary part) of the eigenvalues.
        Return the eigenvectors (each normalised to unit norm) as a list of
        lists of complex numbers.

        :param problem: A non‑symmetric square matrix.
        :return: A list of normalised eigenvectors sorted in descending order.
        """
        # --- 1. Eigen decomposition -------------------------------------------------
        eigenvalues, eigenvectors = np.linalg.eig(problem.astype(complex, copy=False))
        # eigenvectors: columns correspond to eigenvectors

        # --- 2. Get sorting order ---------------------------------------------------
        # Build pairs of (real, imag) for sorting descending.
        real = eigenvalues.real
        imag = eigenvalues.imag
        # argsort on negative values gives descending order.
        order = np.lexsort((-imag, -real))  # first key -real, then -imag
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # --- 3. Normalise eigenvectors -----------------------------------------------
        # Compute norms (Euclidean) for each eigenvector (column).
        norms = np.linalg.norm(eigenvectors, axis=0)
        # Avoid division by zero: any zero norm will be left unchanged (rare).
        nonzero = norms > 1e-12
        eigenvectors[:, nonzero] /= norms[nonzero]

        # --- 4. Convert to Python list ----------------------------------------------
        # Transpose to get list of vectors; then convert each column to a list.
        return [vec.tolist() for vec in eigenvectors.T]