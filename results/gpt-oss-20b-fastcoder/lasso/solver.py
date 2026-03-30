# solver.py
from typing import Any, List
import numpy as np
from sklearn import linear_model


class Solver:
    """Fast implementation of a Lasso regression solver.

    The class uses the efficient coordinate‑descent implementation from
    scikit‑learn, which is typically faster than re‑implementing Lasso
    with plain NumPy.  Minor sanity checks are performed before fitting
    to avoid costly exception handling at runtime.
    """

    # Lasso hyper‑parameters used in all calls
    _alpha = 0.1
    _fit_intercept = False

    def __init__(self):
        # Pre‑create a reusable estimator; scikit‑learn estimators are
        # lightweight enough that re‑instantiation is inexpensive,
        # but having one instance prevents repeated import/initialisation
        # overhead when `solve` is called many times.
        self._estimator = linear_model.Lasso(
            alpha=self._alpha, fit_intercept=self._fit_intercept
        )

    def solve(self, problem: dict[str, Any]) -> List[float]:
        """
        Fit a Lasso model to the data provided in `problem` and return
        the coefficient vector as a Python list.

        Parameters
        ----------
        problem: dict
            Must contain 'X' (2‑D array‑like) and 'y' (1‑D array‑like).

        Returns
        -------
        list[float]
            The coefficients of the fitted Lasso model; if fitting fails
            a zero vector of appropriate dimension is returned.
        """
        X = problem["X"]
        y = problem["y"]

        # Quick sanity check: ensure data is finite.
        if not (
            isinstance(X, np.ndarray)
            or hasattr(X, "__array__")
            or hasattr(X, "toarray")
        ):
            # Not array‑like, return zeros
            return [0.0] * (X.shape[1] if hasattr(X, "shape") else 0)

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # If any NaNs or infs appear, skip fitting
        if not (
            np.isfinite(X_arr).all() and np.isfinite(y_arr).all()
        ):
            d = X_arr.shape[1] if X_arr.ndim == 2 else 0
            return [0.0] * d

        try:
            self._estimator.fit(X_arr, y_arr)
            return self._estimator.coef_.tolist()
        except Exception:
            d = X_arr.shape[1]
            return [0.0] * d