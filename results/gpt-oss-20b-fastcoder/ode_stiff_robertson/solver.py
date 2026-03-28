import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float]) -> Any:
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k1, k2, k3 = problem["k"]

        def rober(t, y):
            y1, y2, y3 = y
            return np.array(
                [
                    -k1 * y1 + k3 * y2 * y3,
                    k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3,
                    k2 * y2 ** 2,
                ],
                dtype=float,
            )

        # Stiff problem: use Radau; no dense output or t_eval to speed up.
        sol = solve_ivp(
            rober,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-12,
            atol=1e-10,
            vectorized=False,
            dense_output=False,
        )
        return sol