from typing import Any, List
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """Return the smallest‑magnitude eigenvalues of a real symmetric sparse matrix.

        The matrix is stored as a SciPy sparse matrix in the ``problem`` dictionary
        under the key ``"matrix"``.  The number of requested eigenvalues ``k`` is
        supplied as a scalar.  This implementation:
        - uses the dense eigendecomposition only for tiny matrices or when
          the requested count is almost the whole spectrum;
        - otherwise falls back to Lanczos via ``scipy.sparse.linalg.eigsh``;
        - guarantees a sorted real result even when ``eigsh`` returns complex values
          due to numerical noise.

        Parameters
        ----------
        problem : dict[str, Any]
            ``{ "matrix": <sparse matrix>, "k": <int> }``

        Returns
        -------
        List[float]
            The $k$ smallest‑magnitude eigenvalues in ascending order.
        """
        mat: sparse.spmatrix = problem["matrix"].asformat("csr")
        k: int = int(problem["k"])
        n: int = mat.shape[0]

        # 1. Tiny matrix or almost full spectrum → dense eigendecomposition
        if k >= n or n < 2 * k + 1:
            vals = np.linalg.eigvalsh(mat.toarray())
            return [float(v) for v in vals[:k]]

        # 2. Sparse Lanczos for the rest
        # ``eigsh`` may raise an exception if the problem is ill‑conditioned;
        # in that case we silently fall back to the dense route.
        try:
            # ``maxiter`` and ``ncv`` are chosen to give a robust but efficient search
            vals = eigsh(
                mat,
                k=k,
                which="SM",  # smallest magnitude
                return_eigenvectors=False,
                maxiter=n * 200,
                ncv=max(2 * k + 1, 20),
            )
        except Exception:
            vals = np.linalg.eigvalsh(mat.toarray())[:k]

        # 3. Sort and strip any complex part resulting from round‑off
        return [float(v.real) for v in np.sort(np.real(vals))]