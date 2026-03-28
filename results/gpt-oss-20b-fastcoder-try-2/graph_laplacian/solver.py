from typing import Any
import numpy as np
import scipy.sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        Compute the graph Laplacian of a weighted graph given in CSR format.
        Supports both normalized and unnormalized Laplacians.

        Returns the Laplacian as a dictionary with CSR components.
        """
        try:
            data = np.asarray(problem['data'], dtype=np.float64)
            indices = np.asarray(problem['indices'], dtype=np.int32)
            indptr = np.asarray(problem['indptr'], dtype=np.int32)
            shape = problem['shape']
            normed = bool(problem.get('normed', False))
            graph_csr = scipy.sparse.csr_matrix(
                (data, indices, indptr), shape=shape, dtype=np.float64
            )
        except Exception as exc:
            # In case of malformed input, return empty CSR
            shape = problem.get('shape', (0, 0))
            return {'laplacian': {'data': [], 'indices': [], 'indptr': [], 'shape': shape}}

        # Compute degree vector: sum of absolute weights in each column
        # For Laplacian we need row sums of absolute values.
        deg = np.abs(graph_csr).sum(axis=1).A1  # shape (n,)

        if normed:
            # Handle zero degrees to avoid division by zero
            inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
            # Build diagonal matrix D^-0.5
            D_inv_sqrt = scipy.sparse.diags(inv_sqrt_deg, format='csr')
            # Normalized Laplacian: I - D^-0.5 * W * D^-0.5
            L = scipy.sparse.eye(graph_csr.shape[0], format='csr') - \
                D_inv_sqrt @ graph_csr @ D_inv_sqrt
        else:
            # Unnormalized: D - W
            D = scipy.sparse.diags(deg, format='csr')
            L = D - graph_csr

        # Ensure CSR format and eliminate explicit zeros
        if not isinstance(L, scipy.sparse.csr_matrix):
            L = L.tocsr()
        L.eliminate_zeros()

        return {
            'laplacian': {
                'data': L.data.tolist(),
                'indices': L.indices.tolist(),
                'indptr': L.indptr.tolist(),
                'shape': L.shape
            }
        }