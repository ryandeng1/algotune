from typing import Any
import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the RBF interpolation problem using scipy.interpolate.RBFInterpolator.

        :param problem: A dictionary containing:
            - "x_train": array-like of training coordinates.
            - "y_train": array-like of training values.
            - "x_test":  array-like of points to evaluate.
            - "rbf_config": dict with RBF parameters ('kernel', 'epsilon', 'smoothing').
        :return: Dictionary with key "y_pred" containing predictions as a list.
        """
        x_train = np.asarray(problem["x_train"], dtype=np.float64, order="C")
        y_train = np.asarray(problem["y_train"], dtype=np.float64, order="C").ravel()
        x_test = np.asarray(problem["x_test"], dtype=np.float64, order="C")

        cfg = problem.get("rbf_config", {})
        rbf_interpolator = RBFInterpolator(
            x_train,
            y_train,
            kernel=cfg.get("kernel"),
            epsilon=cfg.get("epsilon"),
            smoothing=cfg.get("smoothing"),
        )
        y_pred = rbf_interpolator(x_test)
        # Convert to list only at the point of return to keep intermediate
        # data in ndarray format for speed.
        return {"y_pred": y_pred.tolist()}