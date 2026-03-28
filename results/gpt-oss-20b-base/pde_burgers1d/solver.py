import numpy as np
from numba import njit, prange

class Solver:
    @staticmethod
    def solve(problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Fast explicit ODE integration for the 1‑D viscous Burgers equation.
        Uses a single upwind discretisation for the inviscid term and a
        centered second‑order scheme for diffusion.  The integration time
        step is chosen to produce 100 points in the interval [t0, t1].
        """
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # ------------------------------------------------------------------
        # Numerical parameters -------------------------------------------------
        # ------------------------------------------------------------------
        n_points = y0.size                            # spatial grid size
        n_steps  = 99                                 # gives 100 points including t0
        dt = (t1 - t0) / n_steps                      # fixed time step
        dx = params["dx"]
        nu = params["nu"]

        # ------------------------------------------------------------------
        # Pre‑allocate output -------------------------------------------------
        # ------------------------------------------------------------------
        sol = np.empty((n_steps + 1, n_points), dtype=np.float64)
        sol[0] = y0

        # ------------------------------------------------------------------
        # Numba kernel --------------------------------------------------------
        # ------------------------------------------------------------------
        @njit(parallel=True)
        def integrate(u, n_steps, dt, dx, nu, sol):
            for step in prange(n_steps):
                # Upwind advection
                lhs = np.zeros_like(u)
                rhs = np.zeros_like(u)

                # Forward and backward differences with zero padding
                u_pad = np.empty(n_points + 2, dtype=np.float64)
                u_pad[0] = 0.0
                u_pad[1:-1] = u
                u_pad[-1] = 0.0

                dfwd = (u_pad[2:] - u_pad[1:-1]) / dx
                dbwd = (u_pad[1:-1] - u_pad[:-2]) / dx

                lhs = np.where(u >= 0, u * dbwd, u * dfwd)

                # Diffusion (central second derivative)
                d2 = (u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]) / dx ** 2

                rhs = -lhs + nu * d2

                # Explicit Euler step
                u += dt * rhs
                sol[step + 1] = u
            return sol

        # ------------------------------------------------------------------
        # Run integration ----------------------------------------------------
        # ------------------------------------------------------------------
        integrate(y0, n_steps, dt, dx, nu, sol)

        # ------------------------------------------------------------------
        # Return the last time snapshot -------------------------------------
        # ------------------------------------------------------------------
        return {"solution": sol[-1].tolist()}