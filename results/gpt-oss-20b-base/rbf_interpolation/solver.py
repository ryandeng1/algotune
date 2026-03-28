import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem):
        """
        Solve the RBF interpolation problem using scipy.interpolate.RBFInterpolator.
        """
        # Fast numpy conversion
        x_train = np.asarray(problem['x_train'], dtype=float)
        y_train = np.asarray(problem['y_train'], dtype=float).ravel()
        x_test = np.asarray(problem['x_test'], dtype=float)

        # Extract configuration with defaults
        rbf_conf = problem.get('rbf_config', {})
        kernel = rbf_conf.get('kernel', 'linear')
        epsilon = rbf_conf.get('epsilon', None)
        smoothing = rbf_conf.get('smoothing', 0)

        # Build interpolator
        interp = RBFInterpolator(x_train, y_train,
                                 kernel=kernel,
                                 epsilon=epsilon,
                                 smoothing=smoothing)

        # Predict
        y_pred = interp(x_test)

        # Return as Python list (compatible with the original API)
        return {'y_pred': y_pred.tolist()}