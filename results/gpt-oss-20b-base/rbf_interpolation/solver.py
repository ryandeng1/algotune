import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem: dict) -> dict:
        x_train = np.asarray(problem["x_train"], dtype=float)
        y_train = np.asarray(problem["y_train"], dtype=float).ravel()
        x_test = np.asarray(problem["x_test"], dtype=float)

        cfg = problem.get("rbf_config", {})
        kernel = cfg.get("kernel", "linear")
        epsilon = cfg.get("epsilon", None)
        smoothing = cfg.get("smoothing", None)

        rbf = RBFInterpolator(x_train, y_train,
                               kernel=kernel, epsilon=epsilon, smoothing=smoothing)
        y_pred = rbf(x_test)
        return {"y_pred": y_pred.tolist()}