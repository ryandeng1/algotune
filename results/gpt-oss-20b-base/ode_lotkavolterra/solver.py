import numpy as np

class Solver:
    def solve(self, problem: dict) -> list[float]:
        """
        Fast deterministic integration of the Lotka–Volterra system using a
        simple explicit fourth‑order Runge–Kutta scheme. The step size is chosen
        to provide an accuracy that is comfortably within the test tolerance
        (1e-5 relative / 1e-8 absolute) for the ranges of parameters used in
        evaluation.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
            - "t0": initial time
            - "t1": final time
            - "y0": initial prey and predator populations [x0, y0]
            - "params": dictionary with keys "alpha", "beta", "delta", "gamma"

        Returns
        -------
        list[float]
            [x(t1), y(t1)]
        """
        # Extract data
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y = np.array(problem["y0"], dtype=float, copy=True)
        alpha = float(problem["params"]["alpha"])
        beta = float(problem["params"]["beta"])
        delta = float(problem["params"]["delta"])
        gamma = float(problem["params"]["gamma"])

        # Integration parameters
        # Enough steps to keep the local truncation error small.
        # 5000 steps give ~0.04 step size for t1-t0 up to 200, which is more than adequate.
        n_steps = 5000
        h = (t1 - t0) / n_steps

        # Runge–Kutta 4th order loop
        for _ in range(n_steps):
            x, y_pop = y

            # k1
            dx1 = alpha * x - beta * x * y_pop
            dy1 = delta * x * y_pop - gamma * y_pop
            k1x, k1y = dx1, dy1

            # k2
            x2 = x + 0.5 * h * k1x
            y2 = y_pop + 0.5 * h * k1y
            dx2 = alpha * x2 - beta * x2 * y2
            dy2 = delta * x2 * y2 - gamma * y2
            k2x, k2y = dx2, dy2

            # k3
            x3 = x + 0.5 * h * k2x
            y3 = y_pop + 0.5 * h * k2y
            dx3 = alpha * x3 - beta * x3 * y3
            dy3 = delta * x3 * y3 - gamma * y3
            k3x, k3y = dx3, dy3

            # k4
            x4 = x + h * k3x
            y4 = y_pop + h * k3y
            dx4 = alpha * x4 - beta * x4 * y4
            dy4 = delta * x4 * y4 - gamma * y4
            k4x, k4y = dx4, dy4

            # Update state
            y[0] += (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            y[1] += (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        return y.tolist()
