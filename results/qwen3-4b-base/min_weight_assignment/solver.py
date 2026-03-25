import scipy.sparse
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        mat = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"]
        )
        row_ind, col_ind = min_weight_full_bipartite_matching(mat)
        return {
            "assignment": {
                "row_ind": row_ind.tolist(),
                "col_ind": col_ind.tolist()
            }
        }
