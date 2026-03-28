import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem):
        """
        Solves the Perron-Frobenius matrix completion using CVXPY.

        Args:
            problem: Dict containing keys 'inds' (list of 2-element lists),
                     'a'      (list of floats), and 'n' (int).

        Returns:
            dict with estimates 'B' (list of lists of floats) and
            'optimal_value' (float), or None if solving fails.
        """
        n = problem["n"]
        inds = np.asarray(problem["inds"], dtype=int)
        a = np.asarray(problem["a"], dtype=float)

        # All indices of an n×n matrix
        rows = np.repeat(np.arange(n), n)
        cols = np.tile(np.arange(n), n)
        allinds = np.column_stack((rows, cols))

        # Remove the prescribed indices
        mask = np.any((allinds[:, None, :] == inds).all(2), axis=1)
        otherinds = allinds[~mask]

        B = cp.Variable((n, n), pos=True)
        obj = cp.Minimize(cp.pf_eigenvalue(B))
        prod_con = cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0
        assign_con = B[inds[:, 0], inds[:, 1]] == a
        prob = cp.Problem(obj, [prod_con, assign_con])

        try:
            result = prob.solve(gp=True)
        except (cp.SolverError, Exception):
            return None

        if B.value is None:
            return None

        return {"B": B.value.tolist(), "optimal_value": result}