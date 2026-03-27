from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solves the all‑pairs shortest path problem using scipy.sparse.csgraph.shortest_path.
        :param problem: A dictionary representing the graph in CSR components.
        :return: A dictionary with key "distance_matrix":
                 "distance_matrix": The matrix of shortest path distances (list of lists).
                 np.inf is represented as None for JSON serialisation.
        """
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
        except Exception:
            return {"distance_matrix": []}

        try:
            dist_matrix = scipy.sparse.csgraph.shortest_path(
                csgraph=graph_csr, method=self.method, directed=self.directed
            )
        except Exception:
            return {"distance_matrix": []}

        # Replace np.inf with None in a vectorised way
        inf_mask = np.isinf(dist_matrix)
        dist_matrix[inf_mask] = np.nan  # replace by NaN first
        dist_matrix_list = dist_matrix.tolist()
        # Convert NaN to None for JSON compatibility
        for i, row in enumerate(dist_matrix_list):
            for j, val in enumerate(row):
                if val != val:  # NaN test
                    dist_matrix_list[i][j] = None

        return {"distance_matrix": dist_matrix_list}