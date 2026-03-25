import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.array(problem["X"])
        y = np.array(problem["y"])[:, None]
        C = float(problem["C"])
        
        p, n = X.shape[1], X.shape[0]
        
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))
        
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS)
        except cp.SolverError as e:
            return None
        except Exception as e:
            return None
        
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None
        
        beta_value = beta.value.flatten()
        beta0_value = beta0.value
        optimal_value = prob.value
        
        pred = X @ beta_value + beta0_value
        missclass = np.mean((pred * y) < 0)
        
        return {
            "beta0": float(beta0_value),
            "beta": beta_value.tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }
