from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List


def solve(problem: Dict[str, Any]) -> Dict[str, Dict[str, List[int]]]:
    """
    Fast bipartite matching using SciPy's CSR matrix and Hungarian algorithm.

    Parameters
    ----------
    problem : dict
        Should contain:
        * ``data``   : flat array of non‑zero magnitudes.
        * ``indices``: corresponding column indices.
        * ``indptr`` : index pointers for each row.
        * ``shape``  : (n_rows, n_cols).

    Returns
    -------
    dict
        ``{'assignment': {'row_ind': [...], 'col_ind': [...]}}``.
        Empty lists are returned on error.
    """
    try:
        # Convert to NumPy arrays only if they are not already ndarray
        data = np.asarray(problem["data"])
        indices = np.asarray(problem["indices"])
        indptr = np.asarray(problem["indptr"])
        shape = tuple(problem["shape"])

        mat = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=shape
        )
    except Exception:
        return {"assignment": {"row_ind": [], "col_ind": []}}

    try:
        # positional arguments for speed; keyword arguments are slower
        row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(
            mat
        )
    except Exception:
        return {"assignment": {"row_ind": [], "col_ind": []}}

    return {
        "assignment": {
            "row_ind": row_ind.tolist(),
            "col_ind": col_ind.tolist(),
        }
    }