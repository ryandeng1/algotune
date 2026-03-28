import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict[str, object]) -> dict[str, dict[str, list[int]]]:
        # Build CSR matrix (fails silently to empty assignment on error)
        try:
            mat = scipy.sparse.csr_matrix(
                (problem['data'], problem['indices'], problem['indptr']),
                shape=problem['shape']
            )
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Compute minimum weight full bipartite matching
        try:
            row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}"}

        # Convert to plain Python lists for the result
        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}