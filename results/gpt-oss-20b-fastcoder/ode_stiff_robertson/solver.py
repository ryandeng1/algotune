import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem):
        sol = self._solve(problem, debug=False)
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol.y[:, -1].tolist()

    def _solve(self, problem, debug=True):
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        k = tuple(problem["k"])

        def rober(t, y):
            y1, y2, y3 = y
            k1, k2, k3 = k
            return np.array(
                [
                    -k1 * y1 + k3 * y2 * y3,
                    k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3,
                    k2 * y2 ** 2,
                ],
                dtype=float,
            )

        rtol, atol = 1e-11, 1e-9
        method = "Radau"
        if debug:
            t_eval = np.clip(
                np.exp(np.linspace(np.log(1e-6), np.log(t1), 1000)), t0, t1
            )
        else:
            t_eval = None

        return solve_ivp(
            rober,
            [t0, t1],
            y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            dense_output=debug,
        )