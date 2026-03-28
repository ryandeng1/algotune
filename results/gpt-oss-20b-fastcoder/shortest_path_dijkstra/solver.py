from typing import Any, Dict, List
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


class Solver:
    def __init__(self):
        self.directed = False
        self.method = "D"

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Solves the all‑pairs shortest path problem using
        scipy.sparse.csgraph.shortest_path.
        Returns a dictionary with the key ``distance_matrix`` whose value
        is a list of lists (rows). ``None`` represents an infinite distance.
        """
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            dist = scipy.sparse.csgraph.shortest_path(
                csgraph=graph_csr, method=self.method, directed=self.directed
            )
        except Exception:  # pragma: no cover
            return {"distance_matrix": []}

        # Convert to a Python nested list while replacing infinities with None.
        # Using local variables and list comprehensions keeps it fast.
        isinf = np.isinf(dist)
        out = []
        for i in range(dist.shape[0]):
            row = dist[i].tolist()
            for j, val in enumerate(row):
                if isinf[i, j]:
                    row[j] = None
            out.append(row)
        return {"distance_matrix": out}