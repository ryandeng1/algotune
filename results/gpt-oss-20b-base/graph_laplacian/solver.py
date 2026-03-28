import scipy.sparse
import scipy.sparse.csgraph
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Computes the graph Laplacian using scipy.sparse.csgraph.laplacian.

        Parameters
        ----------
        problem : dict
            Keys required:
                * ``data``       – list/array of non‑zero values of the input CSR graph
                * ``indices``    – list/array of column indices
                * ``indptr``     – list/array of index pointers
                * ``shape``      – tuple ``(n, n)``
                * ``normed``     – bool, whether to normalise the Laplacian

        Returns
        -------
        dict
            ``{ 'laplacian': { 'data': [...], 'indices': [...], 'indptr': [...], 'shape': (n, n) } }``

            If the input is invalid or an error occurs the value lists are empty.
        """
        try:
            # Build CSR matrix directly from the input components
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            normed = bool(problem.get("normed", False))

            # Compute Laplacian and convert to CSR if necessary
            L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)
            if not isinstance(L, scipy.sparse.csr_matrix):
                L = L.tocsr()
            L.eliminate_zeros()
            return {
                "laplacian": {
                    "data": L.data.tolist(),
                    "indices": L.indices.tolist(),
                    "indptr": L.indptr.tolist(),
                    "shape": L.shape,
                }
            }
        except Exception:
            # On any failure, return an empty component with the correct shape
            shape = problem.get("shape", (0, 0))
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}