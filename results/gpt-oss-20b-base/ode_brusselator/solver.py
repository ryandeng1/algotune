import numpy as np

class Solver:
    """
    Optimised ODE solver for the Brusselator system.
    Uses a fixed‐step, fourth‑order Runge–Kutta method
    with vectorised numpy operations, avoiding the overhead of
    ``scipy.integrate.solve_ivp``.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol is not None and sol["success"]:
            return sol["y"][:, -1]
        raise RuntimeError(f"Solver failed: {sol.get('message', 'unknown')}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> dict:
        # Initial conditions and parameters
        y0 = np.array(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        A = float(problem["params"]["A"])
        B = float(problem["params"]["B"])

        # Integration settings
        n_steps = 1000 if debug else 200  # trade‑off between speed and accuracy
        h = (t1 - t0) / n_steps

        # Pre‑allocate solution array
        y = np.empty((2, n_steps + 1), dtype=float)
        y[:, 0] = y0
        t = t0

        # Helper for the right‑hand side
        def f(y_vec):
            X, Y = y_vec
            dX_dt = A + X * X * Y - (B + 1) * X
            dY_dt = B * X - X * X * Y
            return np.array([dX_dt, dY_dt], dtype=float)

        # RK4 integration loop
        for i in range(1, n_steps + 1):
            k1 = f(y[:, i - 1])
            k2 = f(y[:, i - 1] + 0.5 * h * k1)
            k3 = f(y[:, i - 1] + 0.5 * h * k2)
            k4 = f(y[:, i - 1] + h * k3)
            y[:, i] = y[:, i - 1] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h

        return {"success": True, "y": y}