# solver.py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class Solver:
    """
    Efficient solver for computing the first k smallest eigenvalues of a symmetric matrix.
    The input matrix is provided in sparse format via the key 'matrix'.
    """

    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Parameters
        ----------
        problem : dict
            Must contain:
            - 'matrix': a scipy sparse matrix (any format)
            - 'k'      : int, number of eigenvalues to return (positive)

        Returns
        -------
        list[float]
            k smallest eigenvalues, sorted in ascending order.
        """
        mat: sparse.spmatrix = problem["matrix"].__class__.toformat(problem["matrix"], "csr")
        k: int = int(problem["k"])
        n = mat.shape[0]

        # Edge case: k >= n or not enough rows for a sparse eigenvalue solve
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # Main case: use sparse eigsh
        # The 'SM' option finds the smallest magnitude eigenvalues
        # ncv is chosen to give a good chance of convergence
        ncv = min(max(2 * k + 1, 20), n - 1)
        try:
            vals = eigsh(
                mat,
                k=k,
                which="SM",
                return_eigenvectors=False,
                ncv=ncv,
                maxiter=n * 200,
            )[0]
        except Exception:
            # Fall back to dense if sparse solver fails
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # Ensure real and sorted
        return [float(v) for v in np.sort(np.real(vals))]