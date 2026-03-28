from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp box problem using CVXPY.

        :param problem: A dictionary of the lp box problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp box problem.
        """
        c = np.array(problem["c"])
        A = np.array(problem["A"])
        b = np.array(problem["b"])
        n = c.shape[0]

        x = cp.Variable(n, lb=0, ub=1)
        prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
        prob.solve(solver="CLARABEL")
        assert prob.status == "optimal"
        return {"solution": x.value.tolist()}