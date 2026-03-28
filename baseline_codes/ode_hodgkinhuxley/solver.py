from typing import Any
import numpy as np
from scipy.integrate import solve_ivp

class Solver:

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.array(problem['y0'])
        t0, t1 = (problem['t0'], problem['t1'])
        params = problem['params']

        def hodgkin_huxley(t, y):
            V, m, h, n = y
            C_m = params['C_m']
            g_Na = params['g_Na']
            g_K = params['g_K']
            g_L = params['g_L']
            E_Na = params['E_Na']
            E_K = params['E_K']
            E_L = params['E_L']
            I_app = params['I_app']
            if V == -40.0:
                alpha_m = 1.0
            else:
                alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            if V == -55.0:
                alpha_n = 0.1
            else:
                alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
            m = np.clip(m, 0.0, 1.0)
            h = np.clip(h, 0.0, 1.0)
            n = np.clip(n, 0.0, 1.0)
            I_Na = g_Na * m ** 3 * h * (V - E_Na)
            I_K = g_K * n ** 4 * (V - E_K)
            I_L = g_L * (V - E_L)
            dVdt = (I_app - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n
            return np.array([dVdt, dmdt, dhdt, dndt])
        rtol = 1e-08
        atol = 1e-08
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(hodgkin_huxley, [t0, t1], y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval, dense_output=debug)
        return sol
