# solver.py
from __future__ import annotations
from typing import Any, Dict

import numpy as np
from scipy.interpolate import RBFInterpolator


class Solver:
    """Fast wrapper for scipy's RBFInterpolator."""

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpolate test points using the configuration supplied in `problem`.

        Parameters
        ----------
        problem : dict
            Must contain 'x_train', 'y_train', 'x_test' and an optional
            'rbf_config' dictionary with keys 'kernel', 'epsilon', 'smoothing'.

        Returns
        -------
        dict
            Evoked values (`y_pred`) stored as a list for JSON serialisation.
        """
        # Fast array views without copying
        x_train = np.asarray(problem["x_train"], dtype=float, order="C")
        y_train = np.asarray(problem["y_train"], dtype=float, order="C").ravel()
        x_test = np.asarray(problem["x_test"], dtype=float, order="C")

        # Default configuration fallback
        rbf_conf = problem.get("rbf_config", {})
        kernel = rbf_conf.get("kernel", "linear")
        epsilon = rbf_conf.get("epsilon", None)
        smoothing = rbf_conf.get("smoothing", 0)

        # Use the most efficient RBFInterpolator
        interpolator = RBFInterpolator(
            x_train, y_train,
            kernel=kernel,
            epsilon=epsilon,
            smoothing=smoothing,
        )

        y_pred = interpolator(x_test)
        return {"y_pred": y_pred.tolist()}