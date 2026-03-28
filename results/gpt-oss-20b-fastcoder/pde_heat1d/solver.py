import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(
        self,
        problem: dict[str, np.ndarray | float],
        debug: bool = True,
    ) -> "scipy.integrate.OptimizeResult":
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        alpha = params["alpha"]
        dx = params["dx"]

        def ode(t, u):
            # Manual padding to avoid repeated np.pad calls
            u_ext = np.empty(u.size + 2, dtype=u.dtype)
            u_ext[0] = 0.0
            u_ext[-1] = 0.0
            u_ext[1:-1] = u
            uxx = (u_ext[2:] - 2 * u_ext[1:-1] + u_ext[:-2]) / dx**2
            return alpha * uxx

        rtol, atol = 1e-6, 1e-6
        method = "RK45"
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            ode,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol