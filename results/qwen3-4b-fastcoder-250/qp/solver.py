import numpy as np
from scipy.optimize import minimize

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        Q = np.asarray(problem["Q"], float)
        q = np.asarray(problem["q"], float)
        G = np.asarray(problem["G"], float)
        h = np.asarray(problem["h"], float)
        A = np.asarray(problem["A"], float)
        b = np.asarray(problem["b"], float)
        n = Q.shape[0]
        
        Q = (Q + Q.T) / 2
        
        x0 = np.zeros(n)
        
        def objective(x):
            return 0.5 * x @ Q @ x + q @ x
        
        cons = [
            {'type': 'ineq', 'fun': lambda x: G @ x - h},
            {'type': 'eq', 'fun': lambda x: A @ x - b}
        ]
        
        res = minimize(
            objective,
            x0,
            method='trust-constr',
            constraints=cons,
            options={'disp': False}
        )
        
        if not res.success:
            raise ValueError("Optimization failed")
        
        return {"solution": res.x.tolist()}
