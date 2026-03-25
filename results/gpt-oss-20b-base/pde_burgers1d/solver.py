import numpy as np
from scipy.integrate import solve_ivp
from typing import Any

class Solver:
    """
    Solver for the 1D Burgers' equation with upwind advection and central
    diffusion using the method of lines.
    """
    def _burgers_ode(self, t: float, u: np.ndarray, params: dict) -> np.ndarray:
        """
        Compute du/dt for the Burgers' equation.

        Parameters
        ----------
        t : float
            Current time (unused, but required by solve_ivp signature)
        u : np.ndarray
            Current solution vector (interior points)
        params : dict
            Dictionary containing 'nu' and 'dx'

        Returns
        -------
        np.ndarray
            Time derivative of u
        """
        nu, dx = params["nu"], params["dx"]
        # Pad with zeros for Dirichlet boundaries
        up = np.pad(u, 1, mode="constant", constant_values=0)

        # Diffusion term (central difference)
        diffusion = (up[2:] - 2 * up[1:-1] + up[:-2]) / (dx ** 2)

        # Advection term with upwind scheme
        u_c = up[1:-1]                      # current value at each grid point
        # Forward difference (for u < 0)
        du_dx_fwd = (up[2:] - up[1:-1]) / dx
        # Backward difference (for u > 0)
        du_dx_bwd = (up[1:-1] - up[:-2]) / dx
        # Piecewise selection
        advection = np.where(u_c >= 0,
                             u_c * du_dx_bwd,
                             u_c * du_dx_fwd)

        # du/dt = -advection + nu * diffusion
        return -advection + nu * diffusion

    def solve(self, problem: dict[str, Any]) -> list[float]:
        """
        Solve the Burgers' equation from t0 to t1.

        Parameters
        ----------
        problem : dict
            Dictionary containing 't0', 't1', 'y0', 'params', and 'x_grid'.
            Only 't0', 't1', 'y0', and 'params' are required.

        Returns
        -------
        list[float]
            Solution vector at time t1 (interior grid points)
        """
        # Extract data
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Time integration with SciPy's RK45 solver
        sol = solve_ivp(
            fun=lambda t, y: self._burgers_ode(t, y, params),
            t_span=(t0, t1),
            y0=y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return final state as list
        return sol.y[:, -1].tolist()
