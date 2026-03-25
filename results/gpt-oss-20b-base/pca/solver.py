import numpy as np


class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute the first `n_components` principal components of the
        centered input `X` using an efficient SVD implementation.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                X (list[list[float]] or np.ndarray) – data matrix (m × n)
                n_components (int) – number of components to retain

        Returns
        -------
        V : np.ndarray
            Array of shape (n_components, n) whose rows form an orthonormal
            basis of the top `n_components` principal subspace.
        """
        # Convert to a NumPy array (supports list-of-lists input)
        X = np.asarray(problem["X"], dtype=float)

        # Center the data (subtract column means)
        Xcentered = X - X.mean(axis=0, keepdims=True)

        # Compute SVD of the centered matrix
        # full_matrices=False gives U of shape (m, k'), S of length k', Vt of (k', n)
        U, _, Vt = np.linalg.svd(Xcentered, full_matrices=False)

        k = problem["n_components"]

        # Take the top k right singular vectors (rows of Vt)
        V = Vt[:k, :]

        return V
