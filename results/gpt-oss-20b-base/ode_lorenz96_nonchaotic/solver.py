import numpy as np
from typing import Any, Dict, List

class Solution:
    def __init__(self, success: bool, y: np.ndarray, message: str = ""):
        self.success = success
        self.y = y if y.ndim == 2 else y.reshape(1, -1)
        self.message = message

class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        F = float(problem["F"])

        def lorenz96(x):
            N = x.shape[0]
            # compute x_{i+1}, x_{i-1}, x_{i-2}
            ip1 = np.roll(x, -1)
            im1 = np.roll(x, 1)
            im2 = np.roll(x, 2)
            return (ip1 - im2) * im1 - x + F

        # integration settings
        if debug:
            n_steps = 999  # gives 1000 evaluation points
        else:
            # choose step so that approx 1000 points, but allow larger steps if t1-t0 small
            n_steps = 999

        h = (t1 - t0) / n_steps
        N = y0.size
        y = np.empty((N, n_steps + 1), dtype=np.float64)
        y[:, 0] = y0

        # Runge–Kutta 4
        for k in range(n_steps):
            x = y[:, k]
            k1 = lorenz96(x)
            k2 = lorenz96(x + 0.5 * h * k1)
            k3 = lorenz96(x + 0.5 * h * k2)
            k4 = lorenz96(x + h * k3)
            y[:, k + 1] = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return Solution(success=True, y=y)