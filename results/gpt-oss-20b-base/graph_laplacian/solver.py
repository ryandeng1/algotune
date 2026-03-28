from typing import Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Compute the graph Laplacian directly: L = D - A.
        ``A`` is the input CSR graph. ``D`` is the diagonal degree matrix.
        The result is returned in CSR format components.
        """
        try:
            # Build CSR matrix from provided data
            A = csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            # Compute node degrees (sum of rows)
            deg = np.ravel(A.sum(axis=1))
            # Build diagonal degree matrix
            D = diags(deg, offsets=0, shape=A.shape, format="csr")
            # Laplacian: L = D - A
            L = D - A
            # Remove explicit zeros
            L.eliminate_zeros()
        except Exception:
            shape = problem.get("shape", (0, 0))
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

        # Return CSR components as lists
        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": L.shape,
            }
        }