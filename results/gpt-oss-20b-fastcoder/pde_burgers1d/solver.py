# solver.py

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, prange
from typing import Dict, Any, Union, List


@njit
def _burgers_step(u: np.ndarray,
                  nu: float,
                  dx: float) -> np.ndarray:
    """
    compute u_t for one time step using 1D Burgers equation with
    zero Dirichlet boundary conditions.
    """
    n = u.size
    # we need u[-1] and u[0] values to compute second order diff
    du_dt = np.empty(n, dtype=u.dtype)

    # first point (boundary - zero)
    du_hat = u[1] - u[0]
    du_prev = 0.0 - u[0]
    u_c = u[0]
    adv = u_c * du_prev
    diff = (u[1] - 2 * u[0] + 0.0) / dx**2
    du_dt[0] = -adv + nu * diff

    # inner points
    for i in range(1, n - 1):
        u_c = u[i]
        du_forward = u[i + 1] - u[i]
        du_backward = u[i] - u[i - 1]
        adv = u_c * (du_backward if u_c >= 0 else du_forward)
        diff = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
        du_dt[i] = -adv + nu * diff

    # last point (boundary - zero)
    du_prev = u[-1] - u[-2]
    du_hat = 0.0 - u[-1]
    u_c = u[-1]
    adv = u_c * du_prev
    diff = (0.0 - 2 * u[-1] + u[-2]) / dx**2
    du_dt[-1] = -adv + nu * diff

    return du_dt


class Solver:
    """
    Optimised solver for 1‑D Burgers equation
    """

    def __init__(self) -> None:
        # nothing special to initialise
        pass

    def solve(self,
              problem: Dict[str, Union[np.ndarray, float]]) -> Dict[str, List[float]]:
        """
        Solve the ODE system defined by `problem` and return the final state.
        """
        sol = self._solve(problem)
        if sol.success:
            return {"y": sol.y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self,
               problem: Dict[str, Union[np.ndarray, float]],
               debug: bool = False) -> Any:
        """
        Core integration routine.
        """
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        nu = params["nu"]
        dx = params["dx"]

        def burgers_equation(t: float, u: np.ndarray) -> np.ndarray:
            # ignore time t
            return _burgers_step(u, nu, dx)

        # use a step‑size controlled solver
        sol = solve_ivp(
            burgers_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e‑6,
            atol=1e‑6,
            t_eval=np.linspace(t0, t1, 100) if debug else None,
            dense_output=False,
        )
        return sol