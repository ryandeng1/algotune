import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    """Highly‑optimised eigenvalue solver for symmetric sparse matrices.

    The implementation decides between a sparse eigensolver (eigsh) and a dense
    routine (numpy.linalg.eigvalsh) depending on the size of the problem and
    the requested number of eigenvalues ``k``.
    """

    @staticmethod
    def _dense_eig(mat: np.ndarray, k: int) -> np.ndarray:
        """Centered dense eigenvalue solver."""
        # ``eigvalsh`` returns all, we slice afterwards
        vals = np.linalg.eigvalsh(mat)
        return vals if k < len(vals) else vals

    @staticmethod
    def _sparse_eig(mat: sparse.spmatrix, k: int) -> np.ndarray:
        """Sparse eigenvalue solver returning the k smallest magnitude eigenvalues."""
        # ``eigsh`` returns the values sorted internally for the requested method
        return eigsh(
            mat,
            k=k,
            which='SM',
            return_eigenvectors=False,
            maxiter=mat.shape[0] * 200,
            ncv=min(mat.shape[0] - 1, max(2 * k + 1, 20)),
        )

    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Parameters
        ----------
        problem : dict
            Must contain:
            - ``matrix`` : a SciPy sparse matrix supporting ``asformat('csr')``.
            - ``k`` : int, desired number of smallest magnitude eigenvalues.

        Returns
        -------
        list[float]
            The requested eigenvalues sorted increasingly.
        """
        mat = problem['matrix'].asformat('csr')
        k = int(problem['k'])
        n = mat.shape[0]

        # If the requested amount is large, fall back to a dense solver
        if k >= n or n < 2 * k + 1:
            vals = self._dense_eig(mat.toarray(), k)
            vals = vals[:k]
        else:
            try:
                vals = self._sparse_eig(mat, k)
            except Exception:
                # Fallback to dense only on failure
                vals = self._dense_eig(mat.toarray(), k)[:k]
            else:
                # eigsh already returns the values in ascending order for 'SM'
                pass

        return [float(v) for v in vals]