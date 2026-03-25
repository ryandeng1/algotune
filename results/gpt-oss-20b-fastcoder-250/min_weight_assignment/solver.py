# solver.py
from typing import Any, Dict, List

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Dict[str, List[int]]]:
        """
        Solve the minimum weight assignment problem for a given sparse cost matrix.

        Parameters
        ----------
        problem : dict
            Dictionary containing the CSR representation of the cost matrix:
                data   : list[float]
                indices: list[int]
                indptr : list[int]
                shape  : list[int] or tuple[int, int]

        Returns
        -------
        dict
            Dictionary with key 'assignment' mapping to another dictionary containing
            'row_ind' and 'col_ind', which are lists of integers representing the
            optimal assignment.
        """
        # Construct CSR matrix from the provided sparse representation
        mat = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
        )

        # Compute minimum weight perfect matching
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)

        # Convert to Python lists before returning
        return {
            "assignment": {
                "row_ind": row_ind.tolist(),
                "col_ind": col_ind.tolist(),
            }
        }
