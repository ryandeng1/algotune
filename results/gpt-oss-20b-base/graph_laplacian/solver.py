import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    """
    Fast solver for computing the graph Laplacian of a CSR graph.
    """

    def solve(self, problem: dict[str, object]) -> dict[str, dict[str, object]]:
        # --- Fast construction of the CSR matrix ---------------------------------
        shape = problem["shape"]
        # Convert input lists to numpy arrays once (fast conversion, no materialisation)
        data = np.asarray(problem["data"], dtype=np.float64)
        indices = np.asarray(problem["indices"], dtype=np.int32)
        indptr = np.asarray(problem["indptr"], dtype=np.int32)

        graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        normed = problem["normed"]

        # --- Compute Laplacian ----------------------------------------------------
        # scipy.sparse.csgraph.laplacian expects a sparse matrix and returns CSR
        L_csr = scipy.sparse.csgraph.laplacian(graph_csr, normed=normed, return_diag=False)
        # Ensure CSR format and strip zeros from the sparse matrix
        L_csr.eliminate_zeros()

        # --- Prepare output -------------------------------------------------------
        lap = {
            "data": L_csr.data.tolist(),
            "indices": L_csr.indices.tolist(),
            "indptr": L_csr.indptr.tolist(),
            "shape": L_csr.shape,
        }
        return {"laplacian": lap}