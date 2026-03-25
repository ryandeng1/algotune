#!/usr/bin/env python3
# solver.py
from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """
    Efficient solver for the 1D viscous Burgers' equation using an
    explicit, stable, low‑overhead integration scheme.
    """

    @staticmethod
    def _burgers_rhs(t: float, u: np.ndarray, nu: float, dx: float) -> np.ndarray:
        """
        Compute the time derivative du/dt for Burgers' equation using
        the method of lines:
            du/dt = -u * du/dx + nu * d^2u/dx^2

        The spatial derivatives are discretised with:
            * upwind differencing for the non–linear advection term
            * central differences for the diffusion term
        Dirichlet boundary conditions (u=0) are imposed by padding.
        """
        # Pad for boundary values
        u_pad = np.pad(u, 1, mode="constant", constant_values=0)

        # Diffusion term (central difference)
        diff = (
            u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]
        ) / (dx * dx)

        # Upwind advection term
        u_center = u_pad[1:-1]  # u_i
        du_dx_fw = (u_pad[2:] - u_pad[1:-1]) / dx  # forward diff
        du_dx_bw = (u_pad[1:-1] - u_pad[:-2]) / dx  # backward diff
        advection = np.where(
            u_center >= 0,
            u_center * du_dx_bw,
            u_center * du_dx_fw,
        )

        # RHS
        return -advection + nu * diff

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the Burgers' equation for the given problem dictionary.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
            - t0: Initial time
            - t1: Final time (fixed at 0.5)
            - y0: Initial spatial profile (list of floats)
            - params: dict with keys 'nu', 'dx', 'num_points'
            - x_grid: list of spatial coordinates (unused in computation)

        Returns
        -------
        list[float]
            Spatial profile at time t1.
        """
        # Extract data
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        nu = problem["params"]["nu"]
        dx = problem["params"]["dx"]

        # Determine a stable explicit timestep
        # CFL condition for advection + diffusion
        u_max = np.max(np.abs(y0))
        dt_adv = dx / max(1e-12, u_max)
        dt_diff = 0.5 * dx * dx / nu if nu > 1e-12 else np.inf
        dt = min(dt_adv, dt_diff)
        # Clamp to keep number of steps reasonable
        nsteps = int(np.ceil((t1 - t0) / dt))
        dt = (t1 - t0) / nsteps

        # Explicit time stepping
        u = y0.copy()
        for _ in range(nsteps):
            du = self._burgers_rhs(t0, u, nu, dx)
            u = u + dt * du

        return u.tolist()
