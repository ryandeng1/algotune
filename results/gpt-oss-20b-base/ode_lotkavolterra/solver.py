import math
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """Return the final state of Lotka–Volterra at t1 using a simple RK4 fixed step."""
        y0 = problem["y0"]
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Parameters
        a = params["alpha"]
        b = params["beta"]
        d = params["delta"]
        g = params["gamma"]

        # Number of RK4 steps
        N = 10000
        h = (t1 - t0) / N

        x, y = y0[0], y0[1]
        for _ in range(N):
            k1x = a * x - b * x * y
            k1y = d * x * y - g * y

            k2x = a * (x + 0.5 * h * k1x) - b * (x + 0.5 * h * k1x) * (y + 0.5 * h * k1y)
            k2y = d * (x + 0.5 * h * k1x) * (y + 0.5 * h * k1y) - g * (y + 0.5 * h * k1y)

            k3x = a * (x + 0.5 * h * k2x) - b * (x + 0.5 * h * k2x) * (y + 0.5 * h * k2y)
            k3y = d * (x + 0.5 * h * k2x) * (y + 0.5 * h * k2y) - g * (y + 0.5 * h * k2y)

            k4x = a * (x + h * k3x) - b * (x + h * k3x) * (y + h * k3y)
            k4y = d * (x + h * k3x) * (y + h * k3y) - g * (y + h * k3y)

            x += (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            y += (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        return {"x": x, "y": y}