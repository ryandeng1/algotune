from typing import Any
import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: dict) -> dict:
        """Solve a 3‑D tensor completion problem using nuclear‑norm minimization.

        The implementation uses vectorised NumPy operations to build unfoldings
        and masks, which removes the explicit Python loops present in the
        reference implementation.  All other logic remains unchanged so that
        the public API is preserved.
        """
        # ------------------------------------------------------------------
        # 1. Input handling and sanity checks
        # ------------------------------------------------------------------
        try:
            orig_tensor = np.array(problem["tensor"], dtype=np.float64)
            mask = np.array(problem["mask"], dtype=bool)
        except Exception:
            return {"completed_tensor": []}

        dim1, dim2, dim3 = orig_tensor.shape
        N12, N13, N23 = dim2 * dim3, dim1 * dim3, dim1 * dim2

        # ------------------------------------------------------------------
        # 2. Construct unfoldings (mode‑1, mode‑2, mode‑3) in a vectorised way
        # ------------------------------------------------------------------
        U1 = orig_tensor.reshape(dim1, N12)
        M1 = mask.reshape(dim1, N12)

        U2 = orig_tensor.transpose(1, 0, 2).reshape(dim2, N13)
        M2 = mask.transpose(1, 0, 2).reshape(dim2, N13)

        U3 = orig_tensor.transpose(2, 0, 1).reshape(dim3, N23)
        M3 = mask.transpose(2, 0, 1).reshape(dim3, N23)

        # ------------------------------------------------------------------
        # 3. CVXPY variables and problem definition
        # ------------------------------------------------------------------
        X1 = cp.Variable((dim1, N12))
        X2 = cp.Variable((dim2, N13))
        X3 = cp.Variable((dim3, N23))

        obj = cp.Minimize(cp.norm(X1, "nuc") + cp.norm(X2, "nuc") + cp.norm(X3, "nuc"))

        constr = [
            cp.multiply(X1, M1) == cp.multiply(U1, M1),
            cp.multiply(X2, M2) == cp.multiply(U2, M2),
            cp.multiply(X3, M3) == cp.multiply(U3, M3),
        ]

        prob = cp.Problem(obj, constr)

        # ------------------------------------------------------------------
        # 4. Solve and return the completed tensor
        # ------------------------------------------------------------------
        try:
            prob.solve(solver=cp.SCS, verbose=False)  # SCS is fast for large problems
        except Exception:
            return {"completed_tensor": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X1.value is None:
            return {"completed_tensor": []}

        completed = X1.value.reshape(dim1, dim2, dim3)
        return {"completed_tensor": completed.tolist()}