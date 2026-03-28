from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Project a real symmetric matrix onto the positive semidefinite cone.

        Parameters
        ----------
        problem : dict
            Dictionary that contains the key 'A', a real symmetric NumPy array.

        Returns
        -------
        dict
            {'X': X} where X is the PSD projection of A.
        """
        A = np.asarray(problem["A"])
        # Eigen-decomposition for symmetric matrices is more efficient and
        # numerically stable than the general eig algorithm.
        eigvals, eigvecs = np.linalg.eigh(A)

        # Clamp negative eigenvalues to zero (within machine precision).
        eigvals = np.maximum(eigvals, 0.0)

        # Reconstruct the matrix.  Using ``@`` avoids allocating the full
        # diagonal matrix and is faster than ``np.diag`` followed by a
        # triple matrix product.
        X = eigvecs @ (eigvals * eigvecs.T)
        return {"X": X}