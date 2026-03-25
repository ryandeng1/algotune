import numpy as np
from scipy.integrate import solve_ivp
from numba import jit

@jit(nopython=True)
def heat_equation_numba(t, u, alpha, dx):
    n = u.shape[0]
    u_padded = np.zeros(n + 2)
    u_padded[0] = 0
    u_padded[1] = 0
    u_padded[2:] = u
    u_padded[-1] = 0
    u_padded[-2] = 0
    u_xx = (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / (dx ** 2)
    return alpha * u_xx

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        y0 = np.array(problem["y0"])
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        alpha = params["alpha"]
        dx = params["dx"]
        n = len(y0)
        
        def heat_equation(t, u):
            return heat_equation_numba(t, u, alpha, dx)
        
        sol = solve_ivp(
            heat_equation,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            t_eval=None,
        )
        
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
