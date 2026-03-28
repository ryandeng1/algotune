from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            source_indices = problem["source_indices"]
            if not isinstance(source_indices, list) or not source_indices:
                raise ValueError("source_indices missing or empty")
        except Exception as e:
            return {"distances": []}

        try:
            dist_matrix = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=source_indices,
                min_only=self.min_only,
            )
        except Exception as e:
            return {"distances": []}

        if dist_matrix.ndim == 1:
            dist_matrix_np = np.array(dist_matrix)
            dist_matrix_np = np.where(np.isinf(dist_matrix_np), None, dist_matrix_np)
            dist_matrix_list = [dist_matrix_np.tolist()]
        else:
            dist_matrix_np = dist_matrix.toarray()
            dist_matrix_np = np.where(np.isinf(dist_matrix_np), None, dist_matrix_np)
            dist_matrix_list = dist_matrix_np.tolist()

        return {"distances": dist_matrix_list}