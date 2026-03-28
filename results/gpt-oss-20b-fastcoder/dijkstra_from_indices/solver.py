import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self):
        self.directed = False
        self.min_only = True

    def solve(self, problem):
        """
        Compute shortest path distances from the given source nodes.
        Returns a nested list with None for unreachable nodes.
        """
        # Build CSR matrix directly
        graph_csr = scipy.sparse.csr_matrix(
            (problem['data'], problem['indices'], problem['indptr']),
            shape=problem['shape']
        )
        source_indices = problem['source_indices']

        # Run Dijkstra
        dist_matrix = scipy.sparse.csgraph.dijkstra(
            csgraph=graph_csr,
            directed=self.directed,
            indices=source_indices,
            min_only=self.min_only
        )

        # Convert to list and replace inf with None
        if dist_matrix.ndim == 1:
            return {'distances': [[None if np.isinf(d) else d for d in dist_matrix]]}
        return {
            'distances': [[None if np.isinf(d) else d for d in row] for row in dist_matrix]
        }