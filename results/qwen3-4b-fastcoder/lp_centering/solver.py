from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        c = np.array(problem["c"])
        A = np.array(problem["A"])
        b = np.array(problem["b"])
        n = c.shape[0]

        x = cp.Variable(n, nonneg=True)
        prob = cp.Problem(cp.Minimize(c.T @ x - cp.sum(cp.log(x))), [A @ x == b])
        prob.solve(solver="CLARABEL")
        assert prob.status == "optimal"
        return {"solution": x.value.tolist()}