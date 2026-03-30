import numpy as np

class Solver:
    """Fast SVD solver that uses NumPy's optimized LAPACK implementation.

    The original implementation used sklearn's `randomized_svd`, which is
    convenient but incurs overhead for small to medium sized dense matrices.
    NumPy's `linalg.svd` performs a truncated SVD via LAPACK and is usually
    faster for the problem sizes encountered in this benchmark.  We keep the
    same public interface and return the left singular vectors `U`, the
    singular values `S`, and the right singular vectors `V` (not transposed).
    """

    def solve(self, problem: dict[str, any]) -> dict[str, list]:
        """Compute a truncated SVD of the matrix `A`.

        Parameters
        ----------
        problem : dict
            Dictionary with the following keys:
                * 'matrix' : 2-D ndarray (dense matrix)
                * 'n_components' : int, number of singular values/vectors to keep
                * 'matrix_type' : str, ignored in this implementation

        Returns
        -------
        dict
            {'U': U, 'S': s, 'V': V}
        """
        A = problem['matrix']
        n_components = problem['n_components']

        # Compute full SVD and truncate. `full_matrices=False` gives compact form.
        # This is typically faster than a randomized algorithm for dense data.
        U, s, Vt = np.linalg.svd(A, full_matrices=False, compute_uv=True)

        # Keep only the requested number of components
        U = U[:, :n_components]
        s = s[:n_components]
        V = Vt[:n_components, :].T

        return {'U': U, 'S': s, 'V': V}