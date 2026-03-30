import numpy as np
from typing import List
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> List[List[complex]]:
        """
        Solve the eigenvector problem for a non‑symmetric square matrix.

        Parameters
        ----------
        problem : NDArray
            Real or complex square matrix of shape (n, n).

        Returns
        -------
        List[List[complex]]
            Normalised eigenvectors, sorted by eigenvalue real part (desc),
            then by imaginary part (desc).  Each inner list is a column
            eigenvector.
        """
        # Compute eigenvalues and right eigenvectors
        vals, vecs = np.linalg.eig(problem)

        # Sorting by real part (desc) then imaginary part (desc)
        # lexsort expects keys in ascending order, so negate to get descending
        key2 = -vals.imag
        key1 = -vals.real
        idx = np.lexsort((key2, key1))

        # Reorder eigenvectors (columns)
        vecs = vecs[:, idx]

        # Normalise columns
        norms = np.linalg.norm(vecs, axis=0)
        # Avoid division by zero (numerical zero eigenvectors)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        # Convert to list of lists of Python complex numbers
        return [vecs[:, i].tolist() for i in range(vecs.shape[1])]