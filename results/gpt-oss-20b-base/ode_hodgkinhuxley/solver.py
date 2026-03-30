# solver.py
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    """
    A fast Hodgkin‑Huxley solver.

    The derivative function is compiled with Numba, eliminating Python overhead for
    every ODE step.  Only the first call builds the Numba code; subsequent calls
    are as fast as pure NumPy.
    """

    def __init__(self):
        # Parameters are extracted once during the first solve to keep the
        # compiled function clean.
        self._compiled_deriv = None

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            # return the final state (V, m, h, n)
            return sol.y[:, -1].tolist()
        raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = False):
        y0 = np.asarray(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']

        # Extract parameters once and forward them to the JIT compiled function.
        C_m   = np.float64(params['C_m'])
        g_Na  = np.float64(params['g_Na'])
        g_K   = np.float64(params['g_K'])
        g_L   = np.float64(params['g_L'])
        E_Na  = np.float64(params['E_Na'])
        E_K   = np.float64(params['E_K'])
        E_L   = np.float64(params['E_L'])
        I_app = np.float64(params['I_app'])

        if self._compiled_deriv is None:
            # define and compile the derivative routine
            @njit
            def deriv(t, y, C_m, g_Na, g_K, g_L,
                      E_Na, E_K, E_L, I_app):
                V, m, h, n = y
                # alpha and beta rates (avoid division by zero)
                # Use conditional expressions to keep numba happy.
                alpha_m = 1.0 if V == -40.0 else 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
                beta_m  = 4.0 * np.exp(-(V + 65.0) / 18.0)
                alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
                beta_h  = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
                alpha_n = 0.1 if V == -55.0 else 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
                beta_n  = 0.125 * np.exp(-(V + 65.0) / 80.0)

                # enforce bounds
                if m < 0.0: m = 0.0
                elif m > 1.0: m = 1.0
                if h < 0.0: h = 0.0
                elif h > 1.0: h = 1.0
                if n < 0.0: n = 0.0
                elif n > 1.0: n = 1.0

                I_Na = g_Na * m ** 3 * h * (V - E_Na)
                I_K  = g_K * n ** 4 * (V - E_K)
                I_L  = g_L * (V - E_L)

                dVdt = (I_app - I_Na - I_K - I_L) / C_m
                dmdt = alpha_m * (1.0 - m) - beta_m * m
                dhdt = alpha_h * (1.0 - h) - beta_h * h
                dndt = alpha_n * (1.0 - n) - beta_n * n

                return np.array([dVdt, dmdt, dhdt, dndt])

            # Wrap the compiled function to provide the constant params
            def _deriv(t, y):
                return deriv(t, y,
                             C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app)

            self._compiled_deriv = _deriv
        else:
            _deriv = self._compiled_deriv

        # Choose integration settings – dense output only when debugging
        rtol = 1e-8
        atol = 1e-8
        method = 'RK45'
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        sol = solve_ivp(
            _deriv, [t0, t1], y0,
            method=method, rtol=rtol, atol=atol,
            t_eval=t_eval, dense_output=debug
        )
        return sol