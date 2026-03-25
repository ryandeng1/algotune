import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = np.array(problem["y0"])
        params = problem["params"]
        nu = params["nu"]
        dx = params["dx"]
        
        @njit
        def burgers_equation_numba(t, u, nu, dx):
            n = u.shape[0]
            u_padded = np.zeros(n + 2)
            u_padded[1:-1] = u
            
            diffusion_term = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / (dx ** 2)
            
            u_centered = u_padded[1:-1]
            du_dx_forward = (u_padded[2:] - u_padded[1:-1]) / dx
            du_dx_backward = (u_padded[1:-1] - u_padded[:-2]) / dx
            
            mask = (u_centered >= 0)
            advection_term = u_centered[mask] * du_dx_backward[mask] + u_centered[~mask] * du_dx_forward[~mask]
            
            return -advection_term + nu * diffusion_term

        def burgers_equation(t, u):
            return burgers_equation_numba(t, u, nu, dx)
        
        sol = solve_ivp(
            burgers_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            t_eval=None
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
