from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]), shape=problem["shape"]
            )
        except Exception as e:
            return {"distance_matrix": []}

        try:
            dist_matrix = scipy.sparse.csgraph.shortest_path(
                csgraph=graph_csr, method=self.method, directed=self.directed
            )
        except Exception as e:
            return {"distance_matrix": []}

        if scipy.sparse.issparse(dist_matrix):
            dist_matrix = dist_matrix.toarray()

        dist_matrix_list = [[None if np.isinf(d) else d for d in row] for row in dist_matrix]

        return {"distance_matrix": dist_matrix_list}