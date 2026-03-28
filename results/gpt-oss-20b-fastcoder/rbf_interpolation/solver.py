from typing import Any
import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the RBF interpolation problem using scipy.interpolate.RBFInterpolator
        with a few micro‑optimisations for speed.
        """
        # Convert data to NumPy arrays with the minimal overhead
        x_train = np.asarray(problem['x_train'], dtype=np.float64)
        y_train = np.asarray(problem['y_train'], dtype=np.float64).ravel()
        x_test  = np.asarray(problem['x_test'],  dtype=np.float64)

        rbf_cfg = problem.get('rbf_config', {})
        # Retrieve kernel parameters directly from the config dictionary
        kernel    = rbf_cfg.get('kernel')
        epsilon   = rbf_cfg.get('epsilon')
        smoothing = rbf_cfg.get('smoothing')

        # Create the interpolator and evaluate it on the test points
        interp = RBFInterpolator(x_train, y_train,
                                 kernel=kernel, epsilon=epsilon, smoothing=smoothing)
        y_pred = interp(x_test)

        # Return a plain Python dict with the predictions as lists
        return {'y_pred': y_pred.tolist()}