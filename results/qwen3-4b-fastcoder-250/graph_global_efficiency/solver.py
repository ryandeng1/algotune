import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        rows = []
        cols = []
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                rows.append(i)
                cols.append(j)
        
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        
        try:
            dist_matrix = shortest_path(graph, directed=False, unweighted=True)
        except:
            dist_matrix = shortest_path(graph, directed=False)
        
        valid_mask = np.isfinite(dist_matrix)
        mask_i_j = np.ones((n, n), dtype=bool)
        mask_i_j[np.arange(n), np.arange(n)] = False
        valid = valid_mask & mask_i_j
        
        total = np.sum(1.0 / dist_matrix[valid])
        global_eff = total / (n * (n - 1))
        return {"global_efficiency": float(global_eff)}
