from typing import Any
import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the Chebyshev center problem using CVXPY.

        :param problem: A dictionary of the Chebyshev center problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the Chebyshev center problem.
        """
        a = np.asarray(problem["a"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        n = a.shape[1]

        x = cp.Variable(n)
        r = cp.Variable()
        constraints = [a @ x + r * cp.norm(a, axis=1) <= b]
        prob = cp.Problem(cp.Maximize(r), constraints)

        # Use the default solver (ECOS) which is fast for small LPs
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Solver failed with status {prob.status}")

        return {"solution": x.value.tolist()}