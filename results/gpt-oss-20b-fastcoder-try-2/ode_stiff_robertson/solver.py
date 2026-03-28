import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# JIT‑compiled RHS for the Robertson ODE
@njit
def rober_jit(t, y, k1, k2, k3):
    y1, y2, y3 = y[0], y[1], y[2]
    f0 = -k1 * y1 + k3 * y2 * y3
    f1 = k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
    f2 = k2 * y2 * y2
    return np.array([f0, f1, f2], dtype=np.float64)


class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol.y[:, -1].tolist()

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True):
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        k1, k2, k3 = tuple(problem["k"])

        # Wrapper to forward constants to the JIT function
        def f(t, y):
            return rober_jit(t, y, k1, k2, k3)

        rtol = 1e-11
        atol = 1e-9
        method = "Radau"

        if debug:
            # 1000 evaluation points spaced exponentially
            val_log = np.linspace(np.log(max(1e-6, t0)), np.log(t1), 1000)
            t_eval = np.exp(val_log)
        else:
            t_eval = None

        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol