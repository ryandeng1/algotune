import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        data = np.array(problem["data"])
        indices = np.array(problem["indices"])
        indptr = np.array(problem["indptr"])
        shape = problem["shape"]
        mat = csr_matrix((data, indices, indptr), shape=shape)
        row_ind, col_ind = min_weight_full_bipartite_matching(mat)
        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}
