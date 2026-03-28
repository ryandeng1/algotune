import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def _burger_step(u, nu, dx):
    """
    Compute the right–hand side of the 1‑D viscous Burgers equation
    for an internal grid with homogeneous Dirichlet boundary conditions.
    """
    N = u.size
    # auxiliary array containing the boundary values (=0)
    u_ext = np.empty(N + 2, dtype=u.dtype)
    u_ext[1:-1] = u
    # difference quotients
    diff_diff = (u_ext[2:] - 2.0 * u_ext[1:-1] + u_ext[:-2]) / (dx * dx)
    diff_forward = (u_ext[2:] - u_ext[1:-1]) / dx
    diff_backward = (u_ext[1:-1] - u_ext[:-2]) / dx

    # advection term (upwind)
    adv = np.empty(N, dtype=u.dtype)
    for i in range(N):
        if u[i] >= 0.0:
            adv[i] = u[i] * diff_backward[i]
        else:
            adv[i] = u[i] * diff_forward[i]

    return -adv + nu * diff_diff


class Solver:

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool) -> Any:
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        nu, dx = problem['params']['nu'], problem['params']['dx']

        def rhs(t, u):
            return _burger_step(u, nu, dx)

        t_eval = np.linspace(t0, t1, 100) if debug else None
        sol = solve_ivp(
            rhs,          # RHS function
            [t0, t1],     # integration interval
            y0,           # initial condition
            method='RK45',
            rtol=1e-6, atol=1e-6,
            t_eval=t_eval,
            dense_output=debug
        )
        return sol