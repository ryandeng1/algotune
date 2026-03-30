# solver.py
import numpy as np
import scipy.sparse
from typing import Any, Dict


class Solver:
    """
    Compute the graph Laplacian of a sparse CSR graph.

    The input format matches the one used by the original reference
    implementation – a dictionary with keys ``data``, ``indices``,
    ``indptr`` and ``shape`` describing a CSR matrix together with
    a boolean ``normed`` flag.

    The solver returns a dictionary containing a sub‑dictionary
    ``"laplacian"`` which stores the CSR components of the result
    using plain Python lists – this keeps the public API identical
    to the reference code while still using efficient NumPy
    arrays for all intermediate calculations.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        try:
            # build the input graph
            data = np.asarray(problem["data"], dtype=np.float64, order="C")
            indices = np.asarray(problem["indices"], dtype=np.int32, order="C")
            indptr = np.asarray(problem["indptr"], dtype=np.int32, order="C")
            shape = tuple(problem["shape"])
            graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

            # compute the Laplacian
            L = scipy.sparse.csgraph.laplacian(
                graph_csr, normed=bool(problem.get("normed", False))
            )
            # ensure we work with a CSR matrix
            if not isinstance(L, scipy.sparse.csr_matrix):
                L = L.tocsr()

            # remove explicit zeros – csr_matrix already does it,
            # so this is a no‑op but kept for API compatibility
            L.eliminate_zeros()
        except Exception:
            # on any error return an empty laplacian with the input shape
            shape = tuple(problem.get("shape", (0, 0)))
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

        # convert the result to plain lists for the public API
        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": L.shape,
            }
        }