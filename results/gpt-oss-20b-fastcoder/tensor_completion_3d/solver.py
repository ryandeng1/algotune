# solver.py
import numpy as np
import cvxpy as cp


class Solver:
    """
    Tensor completion solver using nuclear‑norm minimisation over the three
    mode‑unfoldings.  The implementation avoids Python loops by creating the
    unfolded matrices with vectorised indexing, which is essential for speed
    when the tensor is large.
    """

    def solve(self, problem: dict) -> dict:
        # ------------------------------------------------------------------
        # 1.  Prepare data
        # ------------------------------------------------------------------
        tensor = np.asarray(problem["tensor"], dtype=float)
        mask = np.asarray(problem["mask"], dtype=bool)

        d1, d2, d3 = tensor.shape  # tensor dimensions

        # Forced conversion to 64‑bit float required by cvxpy
        tensor = tensor.astype(np.float64, copy=False)
        mask = mask.astype(bool, copy=False)

        # ----------------------------------------
        # 2.  Construct the three mode‑unfoldings
        # ----------------------------------------
        # mode‑1 unfolding
        U1 = tensor.reshape(d1, -1)
        M1 = mask.reshape(d1, -1)

        # mode‑2 unfolding: use advanced indexing
        # indices for rows (sampled from dim2)
        row_idx = np.repeat(np.arange(d2), d3)
        # indices for columns: (dim1 × dim3) columns
        col_idx = np.arange(d1).repeat(d3) * d3 + np.tile(np.arange(d3), d1)
        U2 = tensor.transpose(1, 0, 2).reshape(d2, -1)
        M2 = mask.transpose(1, 0, 2).reshape(d2, -1)

        # mode‑3 unfolding
        U3 = tensor.transpose(2, 0, 1).reshape(d3, -1)
        M3 = mask.transpose(2, 0, 1).reshape(d3, -1)

        # ------------------------------------------------------------------
        # 3.  CVXPY optimisation
        # ------------------------------------------------------------------
        X1 = cp.Variable(U1.shape, name="X1")
        X2 = cp.Variable(U2.shape, name="X2")
        X3 = cp.Variable(U3.shape, name="X3")

        # objective: sum of nuclear norms of all unfoldings
        obj = cp.Minimize(
            cp.normNuc(X1) + cp.normNuc(X2) + cp.normNuc(X3)
        )

        # constraints: known entries must match the observations
        cons = [
            cp.multiply(X1, M1) == cp.multiply(U1, M1),
            cp.multiply(X2, M2) == cp.multiply(U2, M2),
            cp.multiply(X3, M3) == cp.multiply(U3, M3),
        ]

        problem = cp.Problem(obj, cons)

        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=2000)
        except Exception:
            return {"completed_tensor": []}

        # ------------------------------------------------------------------
        # 4.  Build result tensor
        # ------------------------------------------------------------------
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X1.value is None:
            return {"completed_tensor": []}

        result = X1.value.reshape(tensor.shape)
        return {"completed_tensor": result.tolist()}