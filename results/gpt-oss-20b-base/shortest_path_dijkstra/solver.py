import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self):
        self.directed = False
        self.method = 'D'

    def solve(self, problem):  # type: ignore[override]
        # Build CSR matrix from components
        try:
            shape = problem.get("shape")
            data = problem.get("data")
            indices = problem.get("indices")
            indptr = problem.get("indptr")
            graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        except Exception:
            return {"distance_matrix": []}

        # Compute all‑pairs shortest paths
        try:
            dist = scipy.sparse.csgraph.shortest_path(
                graph_csr, method=self.method, directed=self.directed
            )
        except Exception:
            return {"distance_matrix": []}

        # Convert inf to None efficiently
        inf_mask = np.isinf(dist)
        dist[inf_mask] = np.nan  # temporarily use NaN
        out = dist.tolist()
        # Replace NaN with None in the resulting list of lists
        for i, row in enumerate(out):
            for j, val in enumerate(row):
                if isinstance(val, float) and np.isnan(val):
                    out[i][j] = None

        return {"distance_matrix": out}