import numpy as np
import cvxpy as cp

class Solver:
    """
    Fast Perron–Frobenius matrix completion solver.
    Uses CVXPY with the SCS solver (default for geometric programming) and
    carefully pre‑computes indices to avoid repeated overhead.
    """

    def solve(self, problem: dict[str, list[list[int]] | list[float] | int]) -> dict[str, list[list[float]] | float]:
        # Extract problem data
        inds = np.array(problem['inds'], dtype=int)
        a = np.array(problem['a'], dtype=float)
        n = problem['n']

        # All matrix indices in the order (row, column)
        rows, cols = np.indices((n, n))
        all_inds = np.vstack((rows.ravel(), cols.ravel())).T

        # Indices that are *not* fixed by the input (to be optimized)
        mask_fixed = np.any(all_inds[:, None] == inds, axis=2)
        other_inds = all_inds[~mask_fixed.ravel()]

        # Decision variables
        B = cp.Variable((n, n), pos=True)

        # Objective: minimize the Perron–Frobenius eigenvalue
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # Constraints:
        #   1. Product of all free entries equals 1 (geometric program)
        #   2. Fixed entries equal the provided values
        prod_constraint = cp.prod(B[other_inds[:, 0], other_inds[:, 1]]) == 1.0
        fixed_constraint = B[inds[:, 0], inds[:, 1]] == a
        constraints = [prod_constraint, fixed_constraint]

        # Build the problem
        prob = cp.Problem(objective, constraints)

        # Solve with SCS (fast for large geometric programs)
        try:
            result = prob.solve(solver=cp.SCS, verbose=False, max_iters=4000, eps=1e-6)
        except Exception:
            return None

        # If the problem is infeasible or unbounded return None
        if B.value is None:
            return None

        return {
            'B': B.value.tolist(),
            'optimal_value': result
        }