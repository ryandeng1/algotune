import numpy as np
from scipy.interpolate import RBFInterpolator

class Solver:
    """
    Fast RBF interpolation solver.
    """

    def __call__(self, problem: dict[str, any]) -> dict[str, any]:
        return self.solve(problem)

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Build an RBF interpolator and evaluate it on test points.
        """
        # --- Data preparation --------------------------------------------------
        x_train = np.asarray(problem['x_train'], dtype=np.float64)
        y_train = np.ravel(problem['y_train'], order="C")
        x_test = np.asarray(problem['x_test'], dtype=np.float64)

        # --- Configuration -----------------------------------------------------
        cfg = problem.get('rbf_config', {})
        kernel = cfg.get('kernel', 'linear')
        epsilon = cfg.get('epsilon', None)
        smoothing = cfg.get('smoothing', None)

        # --- Interpolation -----------------------------------------------------
        rbf = RBFInterpolator(
            x_train,
            y_train,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
            # Use the most optimized backend (Numba) if available
            n_threads=None,
        )
        y_pred = rbf(x_test)

        # --- Result packaging --------------------------------------------------
        return {"y_pred": y_pred.tolist()}