import numpy as np
from scipy.interpolate import RBFInterpolator

def solve(problem: dict) -> dict:
    """
    Compute RBF interpolation using scipy's RBFInterpolator.

    Parameters
    ----------
    problem : dict
        Contains keys:
            * 'x_train' : array-like, shape (n_samples, n_features)
            * 'y_train' : array-like, shape (n_samples,) or (n_samples,1)
            * 'x_test'  : array-like, shape (m_samples, n_features)
            * 'rbf_config' : dict with keys
                - 'kernel'    : str, e.g. 'linear', 'multiquadric', etc.
                - 'epsilon'   : float, shape parameter
                - 'smoothing' : float, optional

    Returns
    -------
    dict
        {'y_pred': list of predicted values}
    """
    # Fast conversion to float arrays
    x_train = np.asarray(problem["x_train"], dtype=float)
    y_train = np.asarray(problem["y_train"], dtype=float).ravel()
    x_test = np.asarray(problem["x_test"], dtype=float)

    cfg = problem.get("rbf_config", {})
    kernel = cfg.get("kernel", "linear")
    epsilon = cfg.get("epsilon", 1.0)
    smoothing = cfg.get("smoothing", 0.0)

    # Build interpolator
    rbf = RBFInterpolator(
        x_train,
        y_train,
        kernel=kernel,
        epsilon=epsilon,
        smoothing=smoothing,
    )

    # Evaluate
    y_pred = rbf(x_test)
    # Return as plain list to keep API stable
    return {"y_pred": y_pred.tolist()}