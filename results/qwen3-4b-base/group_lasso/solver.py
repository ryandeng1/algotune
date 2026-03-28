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

        p = X.shape[1] - 1  # number of features
        m = ulabels.shape[0]  # number of unique groups

        group_idx = np.zeros((p, m))
        group_idx[np.arange(p), inverseinds.flatten()] = 1
        not_group_idx = np.logical_not(group_idx)

        sqr_group_sizes = np.sqrt(pjs)

        # --- Define CVXPY Variables ---
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        lba_const = cp.Constant(lba)

        y = y[:, None]

        # --- Define Objective ---
        linear_pred = cp.sum(X[:, 1:] @ beta, 1) + beta0
        logreg = -cp.sum(y * linear_pred) + cp.sum(cp.logistic(linear_pred))
        grouplasso = lba_const * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqr_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)

        # --- Define Constraints ---
        constraints = [beta[not_group_idx] == 0]

        # --- Solve Problem ---
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve()
        except cp.SolverError as e:
            return None
        except Exception as e:
            return None

        # Check solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        if beta.value is None or beta0.value is None:
            return None

        beta = beta.value[np.arange(p), inverseinds.flatten()]

        return {"beta0": beta0.value, "beta": beta.tolist(), "optimal_value": result}