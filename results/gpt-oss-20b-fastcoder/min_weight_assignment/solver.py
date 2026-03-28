import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Dict[str, list[int]]]:
        try:
            # Build a dense cost matrix directly from the sparse data
            n_rows, n_cols = problem['shape']
            data = np.array(problem["data"])
            indices = np.array(problem["indices"])
            indptr = np.array(problem["indptr"])
            # Allocate a dense matrix and fill non‑zero entries
            mat = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
            for row, start, end in zip(range(n_rows), indptr, indptr[1:]):
                mat[row, indices[start:end]] = data[start:end]
            # Handle the special case of an empty graph
            if mat.size == 0:
                return {"assignment": {"row_ind": [], "col_ind": []}}
            # Compute the minimum weight full bipartite matching
            row_ind, col_ind = linear_sum_assignment(mat, maximize=False)
            return {
                "assignment": {
                    "row_ind": row_ind.tolist(),
                    "col_ind": col_ind.tolist()
                }
            }
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}