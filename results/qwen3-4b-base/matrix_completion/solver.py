from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(
        self, problem: dict[str, list[list[int]] | list[float] | int]
    ) -> dict[str, list[list[float]] | float]:
        n = problem["n"]
        inds_list = problem["inds"]
        a_arr = np.array(problem["a"])

        mask = np.zeros((n, n), dtype=bool)
        for i, j in inds_list:
            mask[i, j] = True

        i_indices, j_indices = np.where(~mask)
        otherinds = np.vstack((i_indices, j_indices)).T
        inds_np = np.array(inds_list)

        B = cp.Variable((n, n), pos=True)
        objective = cp.Minimize(cp.pf_eigenvalue(B))
        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds_np[:, 0], inds_np[:, 1]] == a_arr,
        ]

        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except cp.SolverError as e:
            return None
        except Exception as e:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        if B.value is None:
            return None

        return {
            "B": B.value.tolist(),
            "optimal_value": result,
        }