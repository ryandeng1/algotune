import sys
import math
from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        self._y = problem["y0"]
        y0 = self._y.copy()
        t0, t1 = problem["t0"], problem["t1"]
        p = problem["params"]
        A, B = p["A"], p["B"]

        # Simple fourth‑order Runge‑Kutta (fixed step) – fastest for two‑dimensional system
        n_steps = 1000
        h = (t1 - t0) / n_steps
        X, Y = y0
        for _ in range(n_steps):
            d1 = A + X * X * Y - (B + 1) * X
            d2 = B * X - X * X * Y

            X1 = X + 0.5 * h * d1
            Y1 = Y + 0.5 * h * d2
            d1_ = A + X1 * X1 * Y1 - (B + 1) * X1
            d2_ = B * X1 - X1 * X1 * Y1

            X2 = X + 0.5 * h * d1_
            Y2 = Y + 0.5 * h * d2_
            d1__ = A + X2 * X2 * Y2 - (B + 1) * X2
            d2__ = B * X2 - X2 * X2 * Y2

            X3 = X + h * d1__
            Y3 = Y + h * d2__
            d1___ = A + X3 * X3 * Y3 - (B + 1) * X3
            d2___ = B * X3 - X3 * X3 * Y3

            X += (h / 6) * (d1 + 2 * d1_ + 2 * d1__ + d1___)
            Y += (h / 6) * (d2 + 2 * d2_ + 2 * d2__ + d2___)

        return {"X": [X], "Y": [Y]}