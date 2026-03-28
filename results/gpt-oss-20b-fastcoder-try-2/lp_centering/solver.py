import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp centering problem using CVXPY with a fast cone solver.

        :param problem: Dictionary with keys 'c', 'A', 'b'.
        :return: Dictionary containing the optimal solution under key 'solution'.
        """
        # Extract data as NumPy arrays
        c = np.asarray(problem['c'], dtype=np.float64)
        A = np.asarray(problem['A'], dtype=np.float64)
        b = np.asarray(problem['b'], dtype=np.float64)

        n = c.size
        # Positive cone variable to keep log(x) defined
        x = cp.Variable(n, pos=True)

        # Objective: cᵀx - Σ log(xᵢ)
        objective = cp.Minimize(c.T @ x - cp.sum(cp.log(x)))

        # Equality constraints
        constraints = [A @ x == b]

        # Build and solve the problem with a fast interior‑point solver
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False, max_iters=2500, eps_abs=1e-9, eps_rel=1e-9)

        if prob.status not in {'optimal', 'optimal_inaccurate'}:
            raise RuntimeError(f"Solver failed with status '{prob.status}'")

        # Convert solution to Python list
        return {'solution': x.value.tolist()}