import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self, method="FW", directed=False):
        self.method = method
        self.directed = directed

    def solve(self, problem: dict[str, object]) -> dict[str, list[list[float]]]:
        """
        Computes all‑pairs shortest paths for a graph stored as a CSR
        representation.  The result is returned as a list of lists,
        with `None` in place of unreachable nodes (`np.inf`).
        """
        # Build CSR matrix directly from the supplied data
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
        )

        # Compute distances
        dist_matrix = scipy.sparse.csgraph.shortest_path(
            csgraph=graph_csr,
            method=self.method,
            directed=self.directed,
        )

        # Convert to list while replacing infinities with None
        distance_matrix = np.where(
            np.isinf(dist_matrix), None, dist_matrix
        ).tolist()

        return {"distance_matrix": distance_matrix}