from typing import Any
import numpy as np
from scipy.interpolate import RBFInterpolator


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve an RBF interpolation problem with scipy's RBFInterpolator.

        Parameters
        ----------
        problem : dict
            Must contain:
                * "x_train": array-like, shape (n_samples, n_features)
                * "y_train": array-like, shape (n_samples,) or (n_samples,1)
                * "x_test" : array-like, shape (m_samples, n_features)
                * "rbf_config" : dict with keys "kernel", "epsilon", "smoothing"

        Returns
        -------
        dict
            Dictionary with key "y_pred" containing predictions at x_test.
        """
        # Convert inputs to contiguous arrays of dtype float64 for speed
        x_train = np.asarray(problem["x_train"], dtype=np.float64, order="C")
        y_train = np.asarray(problem["y_train"], dtype=np.float64, order="C").ravel()
        x_test = np.asarray(problem["x_test"], dtype=np.float64, order="C")

        # Retrieve configuration and provide defaults if missing
        rbf_cfg = problem.get("rbf_config", {})
        kernel = rbf_cfg.get("kernel", "cubic")
        epsilon = rbf_cfg.get("epsilon", None)
        smoothing = rbf_cfg.get("smoothing", 0.0)

        # Build the interpolator
        interpolator = RBFInterpolator(
            x_train,
            y_train,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
        )

        # Evaluate predictions
        y_pred = interpolator(x_test)

        return {"y_pred": y_pred.tolist()}