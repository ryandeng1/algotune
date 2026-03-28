import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        y0 = np.array(problem['y0'], dtype=np.float64)
        t0, t1 = float(problem['t0']), float(problem['t1'])
        F = float(problem['F'])
        N = y0.size

        # Pre‑compute neighbour indices for the Lorenz‑96 equations
        ip1 = np.roll(np.arange(N, dtype=np.int64), -1)
        im1 = np.roll(np.arange(N, dtype=np.int64), 1)
        im2 = np.roll(np.arange(N, dtype=np.int64), 2)

        # 4th order Runge–Kutta with a fixed step (1000 steps)
        steps = 1000
        h = (t1 - t0) / steps
        x = y0.copy()

        for _ in range(steps):
            k1 = (x[ip1] - x[im2]) * x[im1] - x + F
            k2 = ((x + 0.5 * h * k1)[ip1] - (x + 0.5 * h * k1)[im2]) * \
                 (x + 0.5 * h * k1)[im1] - (x + 0.5 * h * k1) + F
            k3 = ((x + 0.5 * h * k2)[ip1] - (x + 0.5 * h * k2)[im2]) * \
                 (x + 0.5 * h * k2)[im1] - (x + 0.5 * h * k2) + F
            k4 = ((x + h * k3)[ip1] - (x + h * k3)[im2]) * \
                 (x + h * k3)[im1] - (x + h * k3) + F
            x += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0

        return {'y': x.tolist()}