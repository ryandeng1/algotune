from typing import Any
import numpy as np
import scipy.sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Computes the (possibly normalised) graph Laplacian of an undirected
        weighted graph represented in CSR format.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - 'data'   : list[float]   Non‑zero entries of adjacency matrix A
                - 'indices': list[int]     Column indices of non‑zeros
                - 'indptr' : list[int]     Row pointer of CSR
                - 'shape'  : tuple[int,int] Matrix shape (n, n)
                - 'normed' : bool          Normalise column‑wise (`normed=True`)
            The graph is assumed to be undirected, therefore the CSR is symmetric.

        Returns
        -------
        dict
            Dictionary with single key ``'laplacian'`` containing CSR
            components of the Laplacian matrix.
            On failure an empty CSR structure is returned.
        """
        try:
            # Build CSR adjacency matrix
            A = scipy.sparse.csr_matrix(
                (np.asarray(problem["data"], dtype=np.float64),
                 np.asarray(problem["indices"], dtype=np.int32),
                 np.asarray(problem["indptr"], dtype=np.int32)),
                shape=problem["shape"],
            )
            normed = problem.get("normed", False)
        except Exception:
            shape = problem.get("shape", (0, 0))
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

        # Degree vector (sum of rows, graph is undirected so row==col sum)
        deg = np.ravel(A.sum(axis=1))

        if normed:
            # Normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}
            # For sparse matrices, construct D^{-1/2} as a diagonal matrix
            with np.errstate(divide="ignore"):
                inv_sqrt_deg = 1.0 / np.sqrt(deg)
            inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0
            D_inv_sqrt = scipy.sparse.diags(inv_sqrt_deg, format="csr")

            # L = I - D^{-1/2} * A * D^{-1/2}
            L = scipy.sparse.eye(A.shape[0], format="csr", dtype=np.float64) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            # Unnormalised Laplacian: L = D - A
            D = scipy.sparse.diags(deg, format="csr")
            L = D - A

        # Ensure CSR and zero‑out zeros
        L = L.tocsr()
        L.eliminate_zeros()

        # Convert to lists for JSON serialisation
        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": L.shape,
            }
        }