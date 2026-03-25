import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[int]] | list[float] | int]) -> dict[str, list[list[float]] | float]:
        """
        Solves the Perron-Frobenius matrix completion problem using CVXPY.
        """
        inds = np.array(problem["inds"])
        a   = np.array(problem["a"])
        n   = problem["n"]

        xx, yy = np.meshgrid(np.arange(n), np.arange(n))
        allinds = np.vstack((yy.flatten(), xx.flatten())).T
        mask = np.array([(idx not in inds) for idx in allinds], dtype=bool)
        otherinds = allinds[mask]

        B = cp.Variable((n, n), pos=True)

        objective = cp.Minimize(cp.pf_eigenvalue(B))

        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a,
        ]

        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(gp=True)
        except Exception:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or B.value is None:
            return None

        return {
            "B": B.value.tolist(),
            "optimal_value": float(result),
        }
