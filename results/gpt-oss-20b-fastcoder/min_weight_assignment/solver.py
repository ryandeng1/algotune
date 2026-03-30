from typing import Any, Dict, List
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


class Solver:
    """
    Solver for minimum weight full bipartite matching on a sparse matrix.
    The input `problem` dictionary must contain:
        - 'data'   : non‑zero values (1‑D array or list)
        - 'indices': column indices of the non‑zeros (1‑D array or list)
        - 'indptr' : row pointer (1‑D array or list)
        - 'shape'  : (n_rows, n_cols) tuple
    The output is a dictionary with the matching indices.
    """

    @staticmethod
    def solve(problem: Dict[str, Any]) -> Dict[str, Dict[str, List[int]]]:
        # Build CSR matrix.  All conversions are done in one step to avoid
        # temporary Python objects.
        try:
            mat = scipy.sparse.csr_matrix(
                (np.asarray(problem["data"], dtype=np.float64),
                 np.asarray(problem["indices"], dtype=np.int32),
                 np.asarray(problem["indptr"], dtype=np.int32)),
                shape=tuple(problem["shape"]),
            )
        except Exception:
            # If the matrix cannot be constructed, the matching is empty.
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Compute the bipartite matching.  Any exception here also yields
        # an empty assignment.
        try:
            row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Ensure the output is plain Python lists.
        return {
            "assignment": {
                "row_ind": row_ind.tolist(),
                "col_ind": col_ind.tolist(),
            }
        }