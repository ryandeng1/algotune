from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Computes the graph Laplacian using scipy.sparse.csgraph.laplacian.
        Returns CSR components of the Laplacian.
        """
        # Extract CSR data and normed flag
        data: List[float] = problem["data"]
        indices: List[int] = problem["indices"]
        indptr: List[int] = problem["indptr"]
        shape: Tuple[int, int] = problem["shape"]
        normed: bool = problem["normed"]

        # Build CSR matrix
        graph_csr: sp.csr_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)

        # Compute laplacian
        L: sp.csr_matrix = csgraph.laplacian(graph_csr, normed=normed, return_diag=False)

        # Convert to CSR (it may already be)
        if not isinstance(L, sp.csr_matrix):
            L = L.tocsr()

        # eliminate explicit zeros
        L.eliminate_zeros()

        # Prepare result
        res = {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": L.shape,
            }
        }
        return res