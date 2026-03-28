from typing import Any
import numpy as np
from scipy.interpolate import RBFInterpolator


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        x_train = np.array(problem["x_train"], dtype=float)
        y_train = np.array(problem["y_train"], dtype=float).ravel()
        x_test = np.array(problem["x_test"], dtype=float)

        rbf_config = problem.get("rbf_config")
        kernel = rbf_config.get("kernel")
        epsilon = rbf_config.get("epsilon")
        smoothing = rbf_config.get("smoothing")

        rbf_interpolator = RBFInterpolator(
            x_train, y_train, kernel=kernel, epsilon=epsilon, smoothing=smoothing
        )

        y_pred = rbf_interpolator(x_test)

        return {"y_pred": y_pred.tolist()}