import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self):
        self.directed = False
        self.method = "D"

    def solve(self, problem):
        # Construct CSR matrix from triplet format
        csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
        )
        # Compute all‑pairs shortest path matrix
        dist = scipy.sparse.csgraph.shortest_path(
            csr, method=self.method, directed=self.directed
        )
        # Replace infinite values with None for the required output format
        # Use vectorised operations for speed
        inf_mask = np.isinf(dist)
        # Turn the numpy array into list of lists
        mat = dist.tolist()
        for i, row in enumerate(mat):
            if any(inf_mask[i]):  # Only iterate rows containing inf
                row = [
                    None if inf_mask[i][j] else row[j] for j in range(len(row))
                ]
                mat[i] = row
        return {"distance_matrix": mat}