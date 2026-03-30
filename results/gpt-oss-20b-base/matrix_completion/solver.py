"""
solver.py
"""

import numpy as np
import cvxpy as cp

class Solver:
    """
    Solver for the Perron–Frobenius matrix completion problem.

    This implementation focuses on speed:
    * all expensive NumPy operations are performed once and reused
    * the CVXPY problem is built with the minimal amount of pre‑allocation
    *  error handling is lightweight
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any] | None:
        """
        Solve the matrix completion problem.

        Parameters
        ----------
        problem
            Dictionary containing:
            * "inds"  – list of known index pairs (row, col)
            * "a"     – list of known values at these indices
            * "n"     – matrix dimension

        Returns
        -------
        dict | None
            If a solution is found the dictionary contains:
            * "B"             – completed matrix as a list‑of‑lists
            * "optimal_value" – value of the objective (PF eigenvalue)
            If the problem is infeasible or an error occurs, None is returned.
        """

        # ------------------------------------------------------------------
        # Input conversion
        # ------------------------------------------------------------------
        inds = np.asarray(problem["inds"], dtype=np.int64)
        a_val = np.asarray(problem["a"], dtype=np.float64)
        n = int(problem["n"])

        # All index pairs in row-major order
        allinds = np.mgrid[0:n, 0:n].reshape(2, -1).T
        # Mask for known entries
        mask_known = np.isin(allinds, inds, assume_unique=True, invert=True).all(axis=1)
        otherinds = allinds[mask_known]

        # ------------------------------------------------------------------
        # CVXPY model
        # ------------------------------------------------------------------
        B = cp.Variable((n, n), nonneg=True)

        # Objective: minimize PF eigenvalue
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # Constraints
        cons = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a_val,
        ]

        prob = cp.Problem(objective, cons)

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        try:
            result = prob.solve(gp=True)
        except Exception:
            return None

        if B.value is None:
            return None

        return {
            "B": B.value.tolist(),
            "optimal_value": float(result),
        }