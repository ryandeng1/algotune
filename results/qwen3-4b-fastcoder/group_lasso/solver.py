from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(
        self, problem: dict[str, list[list[float]] | list[int] | float]
    ) -> dict[str, list[float] | float]:
        X = np.array(problem["X"])
        y = np.array(problem["y"])
        gl = np.array(problem["gl"])
        lba = problem["lba"]

        ulabels, inverseinds, pjs = np.unique(gl[:, None], return_inverse=True, return_counts=True)
        p = X.shape[1] - 1
        m = ulabels.shape[0]

        sqr_group_sizes = np.sqrt(pjs)

        group_idx = np.zeros((p, m))
        group_idx[np.arange(p), inverseinds.flatten()] = 1
        not_group_idx = np.logical_not(group_idx)

        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        y = y[:, None]

        z = cp.sum(X[:, 1:] @ beta, 1)
        z_plus_beta0 = z[:, np.newaxis] + beta0

        logreg = -cp.sum(y * z_plus_beta0) + cp.sum(cp.logistic(z_plus_beta0))
        grouplasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqr_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)

        constraints = [beta[not_group_idx] == 0]

        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve()
        except cp.SolverError:
            return None
        except Exception:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        if beta.value is None or beta0.value is None:
            return None

        beta = beta.value[np.arange(p), inverseinds.flatten()]

        return {"beta0": beta0.value, "beta": beta.tolist(), "optimal_value": result}