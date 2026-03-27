from typing import Any
import numpy as np
import scipy.sparse as sp


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Computes the graph Laplacian using direct sparse matrix operations.

        :param problem: A dictionary containing CSR components and a `normed` flag.
        :return: A dictionary with key "laplacian" containing CSR components:
                 "data", "indices", "indptr", "shape".
                 Empty components are returned on failure.
        """
        try:
            # Construct adjacency from CSR components
            adj = sp.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            normed = problem["normed"]
        except Exception:
            shape = problem.get("shape", (0, 0))
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

        try:
            n = adj.shape[0]
            # Degree (normed or not)
            if normed:
                # Normalised Laplacian: D^-1/2 * (D - A) * D^-1/2
                degrees = np.array(adj.sum(axis=1)).ravel()
                # Guard against zero degrees
                with np.errstate(divide="ignore", invalid="ignore"):
                    inv_sqrt_deg = 1.0 / np.sqrt(degrees)
                inv_sqrt_deg[np.isnan(inv_sqrt_deg)] = 0.0
                D_half = sp.diags(inv_sqrt_deg)
                L = D_half.dot(adj - sp.diags(degrees)).dot(D_half)
            else:
                # Unnormalised Laplacian: D - A
                degrees = np.array(adj.sum(axis=1)).ravel()
                D = sp.diags(degrees)
                L = D - adj

            L = L.tocsr()
            L.eliminate_zeros()
        except Exception:
            shape = problem["shape"]
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": L.shape,
            }
        }