from typing import Any
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Computes the graph Laplacian using scipy.sparse.csgraph.laplacian.
        Returns the result as a CSR‑formatted dictionary.
        """
        try:
            # Build the sparse matrix once
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            normed = problem.get("normed", False)

            # Compute the Laplacian
            L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)

            # Ensure a CSR matrix and drop zeros
            L_csr = L.tocsr() if not isinstance(L, scipy.sparse.csr_matrix) else L
            L_csr.eliminate_zeros()

            return {
                "laplacian": {
                    "data": L_csr.data.tolist(),
                    "indices": L_csr.indices.tolist(),
                    "indptr": L_csr.indptr.tolist(),
                    "shape": L_csr.shape,
                }
            }
        except Exception:
            # Return an empty CSR structure on failure
            shape = problem.get("shape", (0, 0))
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}