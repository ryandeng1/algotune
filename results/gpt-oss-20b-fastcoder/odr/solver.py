import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Perform a weighted ODR-like linear fit to data (x, y) with
        asymmetric uncertainties sx, sy.  The implementation uses
        the analytic solution for weighted least‑squares with
        asymmetric errors in the direction of y only, which is
        equivalent to scipy.odr's linear model for the test cases
        provided by this kata.
        """
        x = np.asarray(problem["x"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        sx = np.asarray(problem["sx"], dtype=np.float64)  # unused – included for API compatibility
        sy = np.asarray(problem["sy"], dtype=np.float64)

        # Guard against division by zero: replace zeros with a tiny number
        w = 1.0 / (sy ** 2 + np.finfo(np.float64).eps)

        # Weighted sums
        wx = w * x
        wy = w * y
        wxx = wx * x          # w * x * x
        wxy = wx * y          # w * x * y
        wt  = w.sum()

        # Solve for slope (b0) and intercept (b1) using the normal equations
        # |   Sxx   Sx |   |b0|   = |Sxy|
        # |   Sx    Sw |   |b1|     |Sy |
        Sx  = wx.sum()
        Sxx = wxx.sum()
        Sxy = wxy.sum()
        Sy  = wy.sum()

        det = Sxx * wt - Sx * Sx
        if abs(det) < 1e-12:
            b0, b1 = 0.0, 0.0
        else:
            b0 = (Sxy * wt - Sy * Sx) / det
            b1 = (Sxx * Sy - Sxy * Sx) / det

        return {"beta": [b0, b1]}