import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def solve(self, problem: dict) -> dict:
        # Build the CSR matrix once, catching any construction error
        try:
            mat = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
        except Exception:
            # In case of malformed input we return an empty assignment
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Compute the optimal full bipartite matching
        try:
            row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat)
        except Exception:
            return {"assignment": {"row_ind": [], "col_ind": []}}

        # Convert the numpy arrays to lists for the required output format
        return {"assignment": {"row_ind": row_ind.tolist(), "col_ind": col_ind.tolist()}}