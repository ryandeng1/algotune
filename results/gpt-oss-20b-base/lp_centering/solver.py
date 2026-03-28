import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, list]:
        """
        Solve the lp centering problem using CVXPY.

        :param problem: A dictionary of the lp centering problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp centering problem.
        """
        # Convert inputs to numpy arrays (float64) for better performance
        c = np.asarray(problem["c"], dtype=np.float64)
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        n = c.size
        x = cp.Variable(n)

        # Objective: minimize linear term minus sum of logs (strictly concave)
        objective = cp.Minimize(c @ x - cp.sum(cp.log(x)))
        constraints = [A @ x == b]
        prob = cp.Problem(objective, constraints)

        # Use a fast interior-point solver (Cerberus is efficient for this type of problem)
        prob.solve(solver=cp.CERBERUS, verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=500)

        # Standard CVXPY status check
        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"Solver did not find optimal solution: {prob.status}")

        # Return solution as a plain Python list
        return {"solution": x.value.tolist()}