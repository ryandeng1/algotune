from typing import Any
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        except Exception as e:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        try:
            row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
        except Exception as e:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}