import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        c = np.array(problem["c"])
        A = np.array(problem["A"])
        b = np.array(problem["b"])
        n = len(c)
        
        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=[(0, 1)] * n,
            method='highs'
        )
        
        if res.success:
            return {"solution": res.x.tolist()}
        else:
            raise RuntimeError(f"Optimization failed: {res.message}")
