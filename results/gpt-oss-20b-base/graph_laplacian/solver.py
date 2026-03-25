import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Compute the (possibly normalized) graph Laplacian for a sparse CSR graph
        using a highly efficient path with SciPy's csr_matrix and csgraph.laplacian.
        The output is returned in a dictionary containing the CSR components
        required by the benchmark suite.
        """
        try:
            # Construct the CSR representation of the input graph
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=tuple(problem["shape"])
            )
            normed = bool(problem["normed"])
        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": tuple(problem.get("shape", (0, 0)))
                }
            }

        try:
            # Compute the Laplacian (fast SciPy implementation)
            L = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed)

            # Ensure CSR format and drop explicit zeros
            if not isinstance(L, scipy.sparse.csr_matrix):
                L = L.tocsr()
            L.eliminate_zeros()
        except Exception:
            return {
                "laplacian": {
                    "data": [],
                    "indices": [],
                    "indptr": [],
                    "shape": tuple(problem.get("shape", (0, 0)))
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
