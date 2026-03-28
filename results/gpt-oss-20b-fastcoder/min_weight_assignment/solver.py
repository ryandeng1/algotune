from typing import Any, Dict, List
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Dict[str, List[int]]]:
        # build CSR matrix directly without error handling overhead
        mat = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
        )
        # compute full bipartite matching
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
        # convert to plain Python lists once
        return {
            "assignment": {
                "row_ind": row_ind.tolist(),
                "col_ind": col_ind.tolist(),
            }
        }