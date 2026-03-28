from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        a_np = np.array(problem["a"])
        b_np = np.array(problem["b"])
        n = a_np.shape[1]
        
        a = cp.Constant(a_np)
        b = cp.Constant(b_np)
        
        x = cp.Variable(n)
        r = cp.Variable()
        
        norms = cp.norm(a, axis=1)
        
        prob = cp.Problem(cp.Maximize(r), [a @ x + r * norms <= b])
        prob.solve(solver="CLARABEL")
        assert prob.status == "optimal"
        return {"solution": x.value.tolist()}