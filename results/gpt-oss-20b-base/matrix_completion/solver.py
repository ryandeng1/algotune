from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(
        self, problem: dict[str, list[list[int]] | list[float] | int]
    ) -> dict[str, list[list[float]] | float]:
        """
        Solves the Perron-Frobenius matrix completion using CVXPY.

        Args:
            problem: Dict containing inds, a, n.

        Returns:
            Dict with estimates B, optimal_value.
        """
        inds = np.asarray(problem["inds"])
        a = np.asarray(problem["a"])
        n = problem["n"]

        xx, yy = np.mgrid[0:n, 0:n]
        allinds = np.column_stack((yy.ravel(), xx.ravel()))

        # Determine indices that are not given
        mask = np.any(np.all(allinds[:, None] == inds, axis=2), axis=1)
        otherinds = allinds[~mask]

        # CVXPY variables
        B = cp.Variable((n, n), pos=True)

        # Objective: minimize the Perron-Frobenius eigenvalue
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # Constraints
        constraints = [
            cp.prod(cp.as_matrix(B[otherinds[:, 0], otherinds[:, 1]])) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a,
        ]

        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except (cp.SolverError, Exception):
            return None

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or B.value is None:
            return None

        return {
            "B": B.value.tolist(),
            "optimal_value": result,
        }