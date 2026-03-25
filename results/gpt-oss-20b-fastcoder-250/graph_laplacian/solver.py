import numpy as np
import scipy.sparse


class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Compute the graph Laplacian for a CSR-encoded undirected weighted graph.
        If `normed` is False, returns the combinatorial Laplacian L = D - A.
        If `normed` is True, returns the symmetric normalized Laplacian
        L = I - D^{-1/2} A D^{-1/2}.
        The result is returned with CSR components in the same format as the input.
        """
        try:
            data = np.asarray(problem["data"], dtype=np.float64)
            indices = np.asarray(problem["indices"], dtype=np.int32)
            indptr = np.asarray(problem["indptr"], dtype=np.int32)
            shape = tuple(problem["shape"])
            normed = bool(problem["normed"])
        except Exception:
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": (0, 0)}}

        # Build sparse matrix
        try:
            A = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float64)
        except Exception:
            return {"laplacian": {"data": [], "indices": [], "indptr": [], "shape": shape}}

        n = shape[0]
        # Compute degrees
        degrees = np.array(A.sum(axis=1)).reshape(-1)  # sum of weights per row

        if normed:
            # Normalized Laplacian
            # Avoid division by zero for isolated vertices
            inv_sqrt = np.zeros(n, dtype=np.float64)
            nonzero = degrees > 0
            inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

            # Build array of row indices for each non-zero entry
            row_indices = np.repeat(np.arange(n, dtype=np.int32), np.diff(A.indptr))
            col_indices = A.indices
            # Scale data by -inv_sqrt_i * inv_sqrt_j
            scale = inv_sqrt[row_indices] * inv_sqrt[col_indices]
            L_data = -data * scale
            L = scipy.sparse.csr_matrix(
                (L_data, col_indices, A.indptr),
                shape=shape,
                dtype=np.float64,
            )
            # Set diagonal to 1
            L.setdiag(1.0)
        else:
            # Combinatorial Laplacian
            # L = D - A  =>   -A + diag(D)
            L_data = -data
            L = scipy.sparse.csr_matrix(
                (L_data, indices, indptr),
                shape=shape,
                dtype=np.float64,
            )
            # set diagonal to degree
            L.setdiag(degrees)

        L.eliminate_zeros()
        return {
            "laplacian": {
                "data": L.data.tolist(),
                "indices": L.indices.tolist(),
                "indptr": L.indptr.tolist(),
                "shape": list(L.shape),
            }
        }
