# solver.py
import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    """
    Optimized Solver for the Chebyshev center problem.

    The implementation uses CVXPY with a handful of micro‑optimisations:

    * The Euclidean norms of the rows of **a** are pre‑computed with NumPy
      instead of calling `cp.norm` inside the constraint.
    * The constraint is expressed in a single vectorised inequality, avoiding
      the overhead of a Python loop.
    * The solver is left as `CLARABEL` (the default RBFQP solver) to preserve
      the original semantics, but any other QP capable solver could be
      substituted without further changes.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the Chebyshev center problem.

        Parameters
        ----------
        problem : dict[str, Any]
            Dictionary with keys 'a' and 'b' representing the matrix A
            (m × n) and the RHS vector b (m).

        Returns
        -------
        dict[str, list]
            Dictionary with key 'solution' containing the optimal vector x.
        """
        # Convert input to NumPy arrays (fast path for lists/tuples)
        a = np.array(problem['a'], dtype=float, copy=False)
        b = np.array(problem['b'], dtype=float, copy=False)

        # Pre‑compute Euclidean norms of the rows of a
        row_norms = np.linalg.norm(a, axis=1)

        n = a.shape[1]

        # Define CVXPY variables
        x = cp.Variable(n)
        r = cp.Variable()

        # Vectorised constraint: a @ x + r * row_norms <= b
        constraints = [a @ x + r * row_norms <= b]

        # Problem definition and solution
        prob = cp.Problem(cp.Maximize(r), constraints)
        prob.solve(solver='CLARABEL')

        # Ensure the problem solved successfully
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError(f"Solver failed with status {prob.status}")

        # Return solution as plain Python list
        return {'solution': x.value.tolist()}