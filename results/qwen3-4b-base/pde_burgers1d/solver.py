import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        dx = params["dx"]
        nu = params["nu"]
        
        @njit
        def burgers_equation(t, u, dx, nu):
            n = u.shape[0]
            u_padded = np.zeros(n + 2)
            u_padded[1:-1] = u
            
            diffusion_term = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / (dx * dx)
            
            forward_diff = (u_padded[1:] - u_padded[:-1]) / dx
            backward_diff = (u_padded[:-1] - u_padded[:-2]) / dx
            
            mask = (u >= 0)
            advection_term = np.where(mask, u * backward_diff, u * forward_diff)
            
            return -advection_term + nu * diffusion_term
        
        sol = solve_ivp(
            burgers_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            args=(dx, nu)
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
