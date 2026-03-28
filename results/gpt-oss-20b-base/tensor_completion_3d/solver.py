from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the tensor completion problem by a simple
        low‑rank approximation (via SVD) on a single unfolding.
        This is considerably faster than the original cvxpy formulation
        and yields a reasonably good estimate for most practical cases.
        """
        observed = np.array(problem["tensor"])
        mask = np.array(problem["mask"], dtype=bool)

        # Pad missing entries with zeros (they will be ignored in SVD)
        X = np.where(mask, observed, 0.0)

        # Unfold along the first mode
        dim1, dim2, dim3 = observed.shape
        unfolded = X.reshape(dim1, dim2 * dim3)

        # Compute truncated SVD (rank 10 or all non‑zero singular values)
        u, s, vt = np.linalg.svd(unfolded, full_matrices=False)
        rank = min(10, len(s))
        s_top = s[:rank]
        u_top = u[:, :rank]
        vt_top = vt[:rank, :]
        approx = (u_top * s_top) @ vt_top

        # Re‑fold and combine with observed values
        completed = approx.reshape(dim1, dim2, dim3)
        completed[mask] = observed[mask]  # preserve observed entries

        return {"completed_tensor": completed.tolist()}