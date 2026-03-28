from typing import Any, Dict, List
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]]) -> Dict[str, float]:
        """
        Calculates the global efficiency of the graph using SciPy instead of NetworkX
        for better performance on large sparse graphs.

        Args:
            problem: A dictionary containing the adjacency list of the graph.
                     {"adjacency_list": adj_list}

        Returns:
            A dictionary containing the global efficiency.
            {"global_efficiency": efficiency_value}
        """
        adj_list = problem['adjacency_list']
        n = len(adj_list)
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Build adjacency matrix (unweighted, undirected)
        rows, cols = [], []
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                if u < v:          # avoid duplicate edges
                    rows.append(u)
                    cols.append(v)
                    rows.append(v)
                    cols.append(u)
        data = np.ones(len(rows), dtype=np.float64)
        adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

        # Compute shortest path lengths; unweighted graph => use 'unweighted' metric
        dist_matrix = shortest_path(adjacency, directed=False, unweighted=True)

        # Compute global efficiency: average 1/d over all distinct node pairs
        # Ignore infinite distances (disconnected pairs)
        mask = dist_matrix > 0          # distances > 0 (exclude self, keep connected)
        reachable = dist_matrix[mask]
        if reachable.size == 0:
            return {"global_efficiency": 0.0}

        efficiency_sum = np.sum(1.0 / reachable)
        efficiency = efficiency_sum / (n * (n - 1))
        return {"global_efficiency": float(efficiency)}