import numpy as np
from scipy.integrate import solve_ivp
import numba

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        t0 = problem['t0']
        t1 = problem['t1']
        y0 = np.array(problem['y0'])
        params = problem['params']
        
        beta = params['beta']
        sigma = params['sigma']
        gamma = params['gamma']
        omega = params['omega']
        params_tuple = (beta, sigma, gamma, omega)
        
        @numba.njit
        def seirs_numba(t, y, params_tuple):
            S, E, I, R = y
            beta, sigma, gamma, omega = params_tuple
            dSdt = -beta * S * I + omega * R
            dEdt = beta * S * I - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I - omega * R
            return np.array([dSdt, dEdt, dIdt, dRdt])
        
        sol = solve_ivp(
            seirs_numba,
            [t0, t1],
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-10,
            args=(params_tuple,),
            t_eval=None,
            dense_output=False
        )
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol.y[:, -1].tolist()
