import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Perform RBF interpolation using scipy's RBFInterpolator.

        Parameters
        ----------
        problem : dict
            Dictionary containing training data, test points and RBF configuration.

        Returns
        -------
        dict
            Dictionary containing the prediction list and the configuration used.
        """
        # Extract arrays
        x_train = np.asarray(problem["x_train"], dtype=float)
        y_train = np.asarray(problem["y_train"], dtype=float).ravel()
        x_test = np.asarray(problem["x_test"], dtype=float)

        # Extract configuration
        rbf_cfg = problem.get("rbf_config", {})
        kernel = rbf_cfg.get("kernel", "thin_plate_spline")
        epsilon = rbf_cfg.get("epsilon", None)
        smoothing = rbf_cfg.get("smoothing", 0.0)

        # Build interpolator
        interpolator = RBFInterpolator(
            x_train,
            y_train,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
        )

        # Predict on test set
        y_pred = interpolator(x_test)

        return {
            "y_pred": y_pred.tolist(),
            "rbf_config": rbf_cfg,
        }
