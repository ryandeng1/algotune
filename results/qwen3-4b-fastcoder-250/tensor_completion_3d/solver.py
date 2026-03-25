import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        observed_tensor = np.array(problem["tensor"])
        mask = np.array(problem["mask"])
        tensor_dims = observed_tensor.shape
        dim1, dim2, dim3 = tensor_dims
        
        unfolding1 = observed_tensor.reshape(dim1, dim2 * dim3)
        mask1 = mask.reshape(dim1, dim2 * dim3)
        
        unfolding2 = observed_tensor.transpose(1, 0, 2).reshape(dim2, dim1 * dim3)
        mask2 = mask.transpose(1, 0, 2).reshape(dim2, dim1 * dim3)
        
        unfolding3 = observed_tensor.transpose(2, 0, 1).reshape(dim3, dim1 * dim2)
        mask3 = mask.transpose(2, 0, 1).reshape(dim3, dim1 * dim2)
        
        X1 = cp.Variable((dim1, dim2 * dim3))
        X2 = cp.Variable((dim2, dim1 * dim3))
        X3 = cp.Variable((dim3, dim1 * dim2))
        
        objective = cp.Minimize(cp.norm(X1, "nuc") + cp.norm(X2, "nuc") + cp.norm(X3, "nuc"))
        
        constraints = [
            cp.multiply(X1, mask1) == cp.multiply(unfolding1, mask1),
            cp.multiply(X2, mask2) == cp.multiply(unfolding2, mask2),
            cp.multiply(X3, mask3) == cp.multiply(unfolding3, mask3)
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS)
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"completed_tensor": []}
            completed_tensor = X1.value.reshape(tensor_dims)
            return {"completed_tensor": completed_tensor.tolist()}
        except Exception:
            return {"completed_tensor": []}
