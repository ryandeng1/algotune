import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Any, Dict


class Solver:
    """Optimised RBF interpolation solver."""

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an RBF interpolator from training data and predict values
        at test points. The implementation is a lightweight wrapper
        around scipy's RBFInterpolator and is designed for speed and
        numerical stability.

        Parameters
        ----------
        problem : dict
            Must contain the keys 'x_train', 'y_train', 'x_test',
            and 'rbf_config' with the sub‑keys:
            - kernel: str (e.g. "thin_plate_spline")
            - epsilon: float
            - smoothing: float

        Returns
        -------
        dict
            Dictionary with keys:
            - 'y_pred': list of predicted values at test points
            - 'rbf_config': the configuration dictionary that was used
        """
        # Extract data, ensuring efficient conversion to float arrays
        x_train = np.asarray(problem["x_train"], dtype=np.float64)
        y_train = np.asarray(problem["y_train"], dtype=np.float64).ravel()
        x_test = np.asarray(problem["x_test"], dtype=np.float64)

        rbf_cfg = problem["rbf_config"]
        kernel = rbf_cfg["kernel"]
        epsilon = rbf_cfg["epsilon"]
        smoothing = rbf_cfg["smoothing"]

        # Build the interpolator once and reuse for prediction
        interpolator = RBFInterpolator(
            x_train, y_train,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing
        )

        # Predict
        y_pred = interpolator(x_test)

        return {
            "y_pred": y_pred.tolist(),
            "rbf_config": rbf_cfg,
        }
