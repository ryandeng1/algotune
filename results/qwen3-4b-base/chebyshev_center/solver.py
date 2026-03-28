from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        a_np = np.array(problem["a"])
        b_np = np.array(problem["b"])
        m, n = a_np.shape
        norms = np.linalg.norm(a_np, axis=1)
        x = cp.Variable(n)
        r = cp.Variable()
        constraints = [a_np @ x + r * norms <= b_np]
        prob = cp.Problem(cp.Maximize(r), constraints)
        prob.solve(solver="CLARABEL")
        assert prob.status == "optimal"
        return {"solution": x.value.tolist()}