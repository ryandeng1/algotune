from typing import Any
import numpy as np

class Solver:
    """
    Fast weighted linear regression (least‑squares) replacing scipy.odr.
    This uses analytic solution and only depends on NumPy.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Fit a weighted linear model y = b0*x + b1.
        Weights are derived from the y uncertainties only.
        Returns the estimated parameters as a Python list.
        """
        # Prepare data arrays
        x = np.asarray(problem["x"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        sy = np.asarray(problem["sy"], dtype=np.float64)

        # Avoid division by zero in weights
        eps = 1e-15
        w = 1.0 / np.maximum(sy ** 2, eps)

        # Build design matrix
        X = np.column_stack((x, np.ones_like(x)))

        # Weighted least‑squares solution: (XᵀWX)β = XᵀWy
        W = np.diag(w)
        XtW = X.T * w  # (Xᵀ)W
        beta = np.linalg.solve(XtW @ X, XtW @ y)

        return {"beta": beta.tolist()}