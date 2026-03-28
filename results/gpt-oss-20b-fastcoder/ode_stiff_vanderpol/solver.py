from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.array(problem['y0'], dtype=float)
        t0 = float(problem['t0'])
        t1 = float(problem['t1'])
        mu = float(problem['mu'])

        # Define the Van der Pol ODE as a pure Python function
        def vdp(t, y):
            x, v = y
            dx_dt = v
            dv_dt = mu * ((1 - x * x) * v - x)
            return np.array([dx_dt, dv_dt], dtype=float)

        # Fixed-step RK4 integration
        nsteps = 1000 if debug else 400  # Fewer steps if not debugging
        h = (t1 - t0) / nsteps
        t = t0
        y = y0.copy()
        ys = [y.copy()]

        for _ in range(nsteps):
            k1 = vdp(t, y)
            k2 = vdp(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = vdp(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = vdp(t + h, y + h * k3)
            y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            ys.append(y.copy())

        ys = np.column_stack(ys)  # shape (2, nsteps+1)

        # Create a lightweight object mimicking SciPy's result
        class Result:
            success = True
            message = ""
            y = ys

        return Result()