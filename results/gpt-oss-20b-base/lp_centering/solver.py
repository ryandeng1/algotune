from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp centering problem using CVXPY.

        :param problem: A dictionary of the lp centering problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp centering problem.
        """
        # Extract data as NumPy arrays for performance
        c = np.array(problem["c"])
        A = np.array(problem["A"])
        # b is not used directly; CVXPY takes care of broadcasting
        n = c.shape[0]

        # Define the variable
        x = cp.Variable(n, pos=True)

        # Construct and solve the problem
        objective = cp.Minimize(c.T @ x - cp.sum(cp.log(x)))
        constraints = [A @ x == problem["b"]]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)

        # Ensure the problem solved successfully
        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"CVXPY did not find an optimal solution: {prob.status}")

        # Return the solution as a plain Python list
        return {"solution": x.value.tolist()}