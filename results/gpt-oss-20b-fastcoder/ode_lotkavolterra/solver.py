import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solve the Lotka–Volterra system defined in *problem* and return the final state.

        Parameters
        ----------
        problem : dict
            Must contain the keys 'y0', 't0', 't1', and 'params' with the typical
            parameters of the Lotka–Volterra equations.

        Returns
        -------
        dict
            Mapping of variable names to the final time values.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            return dict(zip(["x", "y"], sol.y[:, -1].tolist()))
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = False) -> Any:
        """Internal routine that calls `solve_ivp` with the appropriate arguments."""
        y0 = np.asarray(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        p = problem["params"]

        # Ordinary differential equation for the Lotka–Volterra system.
        def lotka_volterra(t, y):
            x, y = y
            dx_dt = p["alpha"] * x - p["beta"] * x * y
            dy_dt = p["delta"] * x * y - p["gamma"] * y
            return [dx_dt, dy_dt]

        # Solve only the final state; no dense output or intermediate points.
        return solve_ivp(
            lotka_volterra,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-10,
            atol=1e-10,
            t_eval=[t1] if debug else None,
            dense_output=debug,
        )