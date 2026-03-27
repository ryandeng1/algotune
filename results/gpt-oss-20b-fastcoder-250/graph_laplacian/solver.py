from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        # Build CSR matrix from problem
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            normed = problem["normed"]
        except Exception:
            shape = problem.get("shape", (0, 0))
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

        try:
            # Compute row sums for diagonal (degree) matrix
            row_sums = np.asarray(graph_csr.sum(axis=1)).ravel()

            # L = D - A
            L = graph_csr.copy()
            L.data = -L.data  # negate adjacency values
            L = L.tolil()     # easier to set diagonal
            L.setdiag(row_sums)
            L = L.tocsr()
            L.eliminate_zeros()

            if normed:
                # D^{-1/2} L D^{-1/2}
                inv_sqrt = 1.0 / np.sqrt(row_sums)
                D_inv_sqrt = scipy.sparse.diags(inv_sqrt)
                L = D_inv_sqrt @ L @ D_inv_sqrt
                L.eliminate_zeros()

        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": problem.get("shape", (0, 0)),
                }
            }

        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": L.shape,
            }
        }