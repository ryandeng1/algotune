import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    def solve(self, problem):
        # Convert inputs to fast NumPy arrays
        x_train = np.asarray(problem['x_train'], np.float64)
        y_train = np.asarray(problem['y_train'], np.float64).reshape(-1)
        x_test = np.asarray(problem['x_test'], np.float64)

        # Extract RBF configuration
        cfg = problem.get('rbf_config', {})
        kernel = cfg.get('kernel', 'linear')
        epsilon = cfg.get('epsilon', 1.0)
        smoothing = cfg.get('smoothing', 0.0)

        # Build and evaluate the interpolator
        rbf = RBFInterpolator(x_train, y_train,
                              kernel=kernel,
                              epsilon=epsilon,
                              smoothing=smoothing)
        y_pred = rbf(x_test)

        # Return results as lists
        return {'y_pred': y_pred.tolist()}