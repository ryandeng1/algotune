import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List


class Solver:
    """
    Optimised Burgers equation solver using explicit finite‑difference
    discretisation and SciPy's `solve_ivp` with the default
    Runge‑Kutta (RK45) integrator.  The implementation relies entirely
    on NumPy vector operations and avoids the overhead of padding by
    using `np.roll` for one‑dimensional stencil computations.
    """

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """Solve the Burgers equation for the given problem configuration."""
        sol = self._solve(problem, debug=False)
        if sol.success:
            # return the solution at the final time step
            return {"y": sol.y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = False) -> Any:
        """Internal routine that uses SciPy's `solve_ivp`."""
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu, dx = params["nu"], params["dx"]

        # Function that returns the time derivative of the state vector
        def burgers_equation(t, u):
            # Periodic boundaries implemented with np.roll
            u_prev = np.roll(u, 1)      # u_{j-1}
            u_next = np.roll(u, -1)     # u_{j+1}

            # Diffusion term (central difference)
            diffusion_term = (u_next - 2 * u + u_prev) / dx**2

            # Advection term with upwind scheme
            du_dx_forward = (u_next - u) / dx
            du_dx_backward = (u - u_prev) / dx
            advection_term = np.where(u >= 0, u * du_dx_backward, u * du_dx_forward)

            # Right‑hand side of Burgers' equation
            return -advection_term + nu * diffusion_term

        # Solver options
        rtol, atol = 1e-6, 1e-6
        method = "RK45"

        # In debug mode we evaluate the solution at a fine grid for
        # plotting/inspection; otherwise we let the integrator choose
        # its own adaptive step sizes.
        t_eval = np.linspace(t0, t1, 256) if debug else None

        # Call the integrator
        sol = solve_ivp(
            burgers_equation,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol