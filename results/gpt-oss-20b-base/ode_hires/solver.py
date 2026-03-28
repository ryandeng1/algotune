import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem, debug: bool = True):
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        const = np.asarray(problem["constants"], dtype=float)

        def f(t, y):
            c = const
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            f1 = -c[0] * y1 + c[1] * y2 + c[2] * y3 + c[3]
            f2 =  c[0] * y1 - c[4] * y2
            f3 = -c[5] * y3 + c[1] * y4 + c[6] * y5
            f4 =  c[2] * y2 + c[0] * y3 - c[7] * y4
            f5 = -c[8] * y5 + c[1] * y6 + c[1] * y7
            f6 = -c[9] * y6 * y8 + c[10] * y4 + c[0] * y5 - c[1] * y6 + c[10] * y7
            f7 =  c[9] * y6 * y8 - c[11] * y7
            f8 = -c[9] * y6 * y8 + c[11] * y7
            return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=float)

        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            f,
            [t0, t1],
            y0,
            method="Radau",
            rtol=1e-10,
            atol=1e-9,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol