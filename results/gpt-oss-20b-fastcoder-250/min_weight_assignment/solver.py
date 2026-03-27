from typing import Any
import scipy.sparse
import scipy.sparse.csgraph


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
        """Return a minimum weight full bipartite matching for a sparse matrix.

        The input `problem` must contain:
            - ``data``  : 1‑D array of non‑zero values
            - ``indices``: 1‑D array of column indices of the non‑zeros
            - ``indptr`` : 1‑D array that points to row start positions
            - ``shape``  : (rows, cols) tuple describing the matrix shape

        The function builds a ``csr_matrix`` from the COO representation and
        then calls :func:`scipy.sparse.csgraph.min_weight_full_bipartite_matching`.
        If anything goes wrong a fallback empty match is returned.
        """
        try:
            mat = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
            return {
                "assignment": {
                    "row_ind": row_ind.tolist(),
                    "col_ind": col_ind.tolist(),
                }
            }
        except Exception:
            # In case of any error (e.g. bad input shape) return an empty assignment.
            return {"assignment": {"row_ind": [], "col_ind": []}}