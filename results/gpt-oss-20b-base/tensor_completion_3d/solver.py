# solver.py
import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: dict) -> dict:
        """Tensor completion via sum of nuclear norms of unfoldings."""
        # Extract data
        observed_tensor = np.asarray(problem["tensor"], dtype=np.float64)
        mask = np.asarray(problem["mask"], dtype=bool)

        # Dimensions
        dim1, dim2, dim3 = observed_tensor.shape

        # Unfoldings
        # Mode-1: (dim1, dim2*dim3)
        unfolding1 = observed_tensor.reshape(dim1, dim2 * dim3)
        mask1 = mask.reshape(dim1, dim2 * dim3)

        # Mode-2: (dim2, dim1*dim3) – use moveaxis for efficient reshape
        unfolding2 = np.moveaxis(observed_tensor, 0, 1).reshape(dim2, dim1 * dim3)
        mask2 = np.moveaxis(mask, 0, 1).reshape(dim2, dim1 * dim3)

        # Mode-3: (dim3, dim1*dim2)
        unfolding3 = np.moveaxis(observed_tensor, (0, 1), (2, 0)).reshape(dim3, dim1 * dim2)
        mask3 = np.moveaxis(mask, (0, 1), (2, 0)).reshape(dim3, dim1 * dim2)

        # Variables
        X1 = cp.Variable((dim1, dim2 * dim3))
        X2 = cp.Variable((dim2, dim1 * dim3))
        X3 = cp.Variable((dim3, dim1 * dim2))

        # Constraints: match observed entries
        constraints = [
            cp.multiply(X1, mask1) == cp.multiply(unfolding1, mask1),
            cp.multiply(X2, mask2) == cp.multiply(unfolding2, mask2),
            cp.multiply(X3, mask3) == cp.multiply(unfolding3, mask3),
        ]

        # Objective: minimize sum of nuclear norms
        objective = cp.Minimize(
            cp.norm(X1, "nuc") + cp.norm(X2, "nuc") + cp.norm(X3, "nuc")
        )

        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False, max_iters=2000)
        except Exception:
            # Fallback to default solver
            try:
                prob.solve()
            except Exception:
                return {"completed_tensor": []}

        # Check solution validity
        if prob.status not in {"optimal", "optimal_inaccurate"} or X1.value is None:
            return {"completed_tensor": []}

        # Reshape first unfolding back to tensor
        completed = X1.value.reshape(dim1, dim2, dim3)

        return {"completed_tensor": completed.tolist()}
