from typing import Any, Dict, List
import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Dict[str, List[int]]]:
        """
        Compute a minimum weight full bipartite matching for a given sparse cost
        matrix in CSR format.

        Parameters
        ----------
        problem : dict
            Dictionary containing keys "data", "indices", "indptr", and "shape"
            describing the CSR representation of the cost matrix.

        Returns
        -------
        dict
            Dictionary with key "assignment" mapping to a dictionary containing
            two lists: "row_ind" and "col_ind".  The i-th element of these lists
            indicates that row i is matched to column col_ind[i].  Both lists
            are permutations of 0..n-1, where n is the matrix size.
        """
        # Build the sparse matrix from CSR components
        try:
            mat = scipy.sparse.csr_matrix(
                (problem["data"],
                 problem["indices"],
                 problem["indptr"]),
                shape=problem["shape"],
            )
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Compute the minimum weight full bipartite matching
        try:
            row_ind, col_ind = min_weight_full_bipartite_matching(mat)
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Convert to Python lists for JSON serialisation
        return {
            "assignment": {
                "row_ind": row_ind.tolist(),
                "col_ind": col_ind.tolist(),
            }
        }
