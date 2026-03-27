import numpy as np
from scipy.interpolate import RBFInterpolator


class Solver:
    def solve(self, problem: dict) -> dict:
        # Extract numpy arrays once
        x_train = np.asarray(problem["x_train"], dtype=np.float64)
        y_train = np.asarray(problem["y_train"], dtype=np.float64).ravel()
        x_test = np.asarray(problem["x_test"], dtype=np.float64)

        # Unpack RBF configuration – default to None if missing
        rbf_cfg = problem.get("rbf_config", {})
        kernel = rbf_cfg.get("kernel")
        epsilon = rbf_cfg.get("epsilon")
        smoothing = rbf_cfg.get("smoothing")

        # Build interpolator
        interpolator = RBFInterpolator(x_train, y_train,
                                       kernel=kernel,
                                       epsilon=epsilon,
                                       smoothing=smoothing)

        # Predict in one call – vectorised
        y_pred = interpolator(x_test)

        # Convert to Python list for JSON serialisation
        return {"y_pred": y_pred.tolist()}