import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self):
        self.directed = False
        self.method = 'D'

    def solve(self, problem: dict) -> dict[str, list[list[float]]]:
        """
        Solves all‑pairs shortest paths for a graph provided in CSR components.
        Returns the distance matrix as a list of lists, with ``None`` for
        unreachable pairs.
        """
        # Build CSR matrix directly
        csr = scipy.sparse.csr_matrix(
            (problem['data'], problem['indices'], problem['indptr']),
            shape=problem['shape']
        )

        # Compute shortest path distances
        dist = scipy.sparse.csgraph.shortest_path(
            csgraph=csr,
            method=self.method,
            directed=self.directed
        )

        # Convert to nested Python lists, replacing np.inf with None
        dist_obj = np.where(np.isinf(dist), None, dist).tolist()

        return {'distance_matrix': dist_obj}