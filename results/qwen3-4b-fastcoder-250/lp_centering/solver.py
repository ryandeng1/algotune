import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        c = np.array(problem["c"])
        A = np.array(problem["A"])
        b = np.array(problem["b"])
        n = c.shape[0]
        m = A.shape[0]
        
        x = np.ones(n)
        lambda_ = np.zeros(m)
        
        max_iter = 100
        tolerance = 1e-8
        
        for _ in range(max_iter):
            grad_residual = c - 1.0 / x + np.dot(A.T, lambda_)
            constraint_residual = A @ x - b
            
            jacobian_x = np.diag(1.0 / x**2)
            jacobian_lambda = A.T
            
            J = np.vstack([
                np.hstack([jacobian_x, jacobian_lambda]),
                A
            ])
            
            rhs = -np.vstack([grad_residual, constraint_residual])
            
            J_sol = np.linalg.solve(J, rhs)
            dx = J_sol[:n]
            dlambda = J_sol[n:]
            
            x += dx
            lambda_ += dlambda
            
            if np.linalg.norm(dx) < tolerance and np.linalg.norm(dlambda) < tolerance:
                break
                
        return {"solution": x.tolist()}
