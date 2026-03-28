from typing import Any
import numpy as np

class Solver:

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol["success"]:
            return sol["y"][:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol['message']}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> dict[str, Any]:
        y0 = np.array(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # number of steps
        N = 1000 if debug else 200  # adjust if needed for speed/accuracy trade‑off
        h = (t1 - t0) / N

        # pre‑allocate state array
        y = np.empty((2, N + 1), dtype=float)
        y[:, 0] = y0

        a, b, c, I = params["a"], params["b"], params["c"], params["I"]

        def f(v, w):
            dv = v - v**3 / 3 - w + I
            dw = a * (b * v - c * w)
            return dv, dw

        # simple RK4 loop
        for i in range(N):
            v, w = y[:, i]
            k1v, k1w = f(v, w)
            k2v, k2w = f(v + 0.5 * h * k1v, w + 0.5 * h * k1w)
            k3v, k3w = f(v + 0.5 * h * k2v, w + 0.5 * h * k2w)
            k4v, k4w = f(v + h * k3v, w + h * k3w)
            y[0, i + 1] = v + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
            y[1, i + 1] = w + (h / 6) * (k1w + 2 * k2w + 2 * k3w + k4w)

        return {"success": True, "y": y, "message": "Solved"}