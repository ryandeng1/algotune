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
            problem: Dict containing inds, a.

        Returns:
            Dict with estimates B, optimal_value.
        """
        inds = np.array(problem["inds"])
        a = np.array(problem["a"])
        n = problem["n"]

        # Create a mask for known indices
        known_mask = np.zeros((n, n), dtype=bool)
        for r, c in inds:
            known_mask[r, c] = True

        # Get indices where mask is False
        rows, cols = np.where(~known_mask)
        otherinds = np.vstack((rows, cols)).T

        # --- Define CVXPY Variables ---
        B = cp.Variable((n, n), pos=True)

        # --- Define Objective ---
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # --- Define Constraints ---
        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a,
        ]

        # --- Solve Problem ---
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except cp.SolverError as e:
            return None
        except Exception as e:
            return None

        # Check solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        if B.value is None:
            return None

        return {
            "B": B.value.tolist(),
            "optimal_value": result,
        }