import numpy as np
from scipy.integrate import solve_ivp
import numba as nb

# JIT‑compiled RHS for Lorenz ’96
@nb.njit
def lorenz96_numba(t, x, F, N):
    """
    x: 1‑D numpy array of length N
    return dxdt as numpy array
    """
    dxdt = np.empty_like(x)
    # periodic indices
    for i in range(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        dxdt[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dxdt

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = True):
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = float(problem['t0']), float(problem['t1'])
        F = float(problem['F'])
        N = y0.size

        # Wrapper to satisfy solve_ivp signature
        def rhs(t, x):
            return lorenz96_numba(t, x, F, N)

        rtol = 1e-8
        atol = 1e-8
        method = 'RK45'  # use the fast KDOP integrator

        # Only evaluate at intermediate points if debugging
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        return solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )