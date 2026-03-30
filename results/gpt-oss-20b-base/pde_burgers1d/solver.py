from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    A fast solver for the 1‑D viscous Burgers equation.
    The main optimisation is that the RHS is vectorised and the padding
    is performed only once per integration step using a temporary
    zero‑padded array.  All constants are pre‑computed.
    """

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True) -> Any:
        y0: np.ndarray = np.asarray(problem["y0"])
        t0: float = problem["t0"]
        t1: float = problem["t1"]
        params: dict = problem["params"]

        # Pre‑compute constants
        nu: float = params["nu"]
        dx: float = params["dx"]
        inv_dx: float = 1.0 / dx
        inv_dx2: float = inv_dx * inv_dx
        n: int = y0.size

        # Temporary array for padding
        u_pad: np.ndarray = np.empty(n + 2, dtype=y0.dtype)

        def burgers_eq(t: float, u: np.ndarray) -> np.ndarray:
            # Pad with zeros on both sides
            u_pad[0] = u_pad[-1] = 0.0
            u_pad[1:-1] = u

            # Central, forward and backward differences
            center = u_pad[1:-1]
            diff_forward = (u_pad[2:] - center) * inv_dx
            diff_backward = (center - u_pad[:-2]) * inv_dx

            # Advection term with upwind choice
            advection = np.where(center >= 0, center * diff_backward, center * diff_forward)

            # Diffusion term
            diffusion = (u_pad[2:] - 2 * center + u_pad[:-2]) * inv_dx2

            return -advection + nu * diffusion

        # Solver options
        rtol = 1e-06
        atol = 1e-06
        method = "RK45"
        # Use t_eval only for debugging to avoid extra evaluations
        t_eval = np.linspace(t0, t1, 100) if debug else None

        sol = solve_ivp(
            burgers_eq,
            [t0, t1],
            y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            dense_output=debug,
        )
        return sol