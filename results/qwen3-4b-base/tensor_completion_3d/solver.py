import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        observed_tensor = np.array(problem["tensor"])
        mask = np.array(problem["mask"])
        tensor_dims = observed_tensor.shape
        d1, d2, d3 = tensor_dims
        
        # Unfold mode 1: (d1, d2*d3)
        unfolding1 = observed_tensor.reshape(d1, d2 * d3)
        mask1 = mask.reshape(d1, d2 * d3)
        
        # Unfold mode 2: (d2, d1*d3)
        unfolding2 = observed_tensor.transpose(1, 2, 0).reshape(d2, d1 * d3)
        mask2 = mask.transpose(1, 2, 0).reshape(d2, d1 * d3)
        
        # Unfold mode 3: (d3, d1*d2)
        unfolding3 = observed_tensor.transpose(2, 0, 1).reshape(d3, d1 * d2)
        mask3 = mask.transpose(2, 0, 1).reshape(d3, d1 * d2)
        
        # Create variables
        X1 = cp.Variable((d1, d2 * d3))
        X2 = cp.Variable((d2, d1 * d3))
        X3 = cp.Variable((d3, d1 * d2))
        
        # Objective: minimize sum of nuclear norms
        objective = cp.Minimize(cp.norm(X1, "nuc") + cp.norm(X2, "nuc") + cp.norm(X3, "nuc"))
        
        # Constraints
        constraints = [
            cp.multiply(X1, mask1) == cp.multiply(unfolding1, mask1),
            cp.multiply(X2, mask2) == cp.multiply(unfolding2, mask2),
            cp.multiply(X3, mask3) == cp.multiply(unfolding3, mask3),
        ]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or X1.value is None:
                return {"completed_tensor": []}
            completed_tensor = X1.value.reshape(tensor_dims)
            return {"completed_tensor": completed_tensor.tolist()}
        except cp.SolverError:
            return {"completed_tensor": []}
        except Exception:
            return {"completed_tensor": []}
