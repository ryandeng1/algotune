# solver.py
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    """
    Optimized solver for bipartite matching using SciPy sparse matrix routines.
    """

    def solve(self, problem: dict[str, any]) -> dict[str, dict[str, list[int]]]:
        """
        Compute a minimum‑weight perfect matching for a bipartite graph.

        Parameters
        ----------
        problem : dict
            Must contain the keys:
                'data'   – weight values (list or array)
                'indices' – column indices (list or array)
                'indptr'  – index pointers (list or array)
                'shape'   – tuple (n_rows, n_cols)

        Returns
        -------
        dict
            Dictionary with the matching indices converted to Python lists.
        """
        # Build the CSR matrix once; SciPy handles the conversion efficiently
        mat = scipy.sparse.csr_matrix(
            (np.asarray(problem["data"], dtype=np.float64),
             np.asarray(problem["indices"], dtype=np.int32),
             np.asarray(problem["indptr"], dtype=np.int32)),
            shape=tuple(map(int, problem["shape"]))
        )

        # Call the efficient SciPy implementation for bipartite matching
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)

        # Convert to Python lists for the required return format
        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}