import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Optimised Hodgkin‑Huxley solver.
    The heavy lifting is in a purely NumPy function with parameters extracted once.
    Only the final state is requested, so the integration is performed directly to t1.
    """

    def __init__(self):
        # Extract constant parameters from problem dict keys to avoid repeated dictionary lookups
        self.param_keys = [
            'C_m', 'g_Na', 'g_K', 'g_L',
            'E_Na', 'E_K', 'E_L', 'I_app'
        ]

    def _create_rhs(self, params):
        """Create a vectorised RHS function with constants bound."""
        C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app = (params[k] for k in self.param_keys)

        def rhs(t, y):
            V, m, h, n = y

            # gated currents
            alpha_m = 1.0 if V == -40.0 else 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m   = 4.0 * np.exp(-(V + 65.0) / 18.0)
            alpha_h  = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h   = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            alpha_n  = 0.1 if V == -55.0 else 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n   = 0.125 * np.exp(-(V + 65.0) / 80.0)

            # clamp gating variables
            m = np.clip(m, 0.0, 1.0)
            h = np.clip(h, 0.0, 1.0)
            n = np.clip(n, 0.0, 1.0)

            I_Na = g_Na * m**3 * h * (V - E_Na)
            I_K  = g_K  * n**4 * (V - E_K)
            I_L  = g_L * (V - E_L)

            dVdt = (I_app - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n

            return np.array([dVdt, dmdt, dhdt, dndt])

        return rhs

    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """Integrate Hodgkin‑Huxley equation from t0 to t1 and return final state."""
        y0 = np.asarray(problem['y0'], dtype=float)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']

        rhs = self._create_rhs(params)

        # Use a stiff solver (BDF) which is faster for the typical HH equations
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method='BDF',
            rtol=1e-8,
            atol=1e-8,
            t_eval=[t1],      # only need final value
        )

        if not sol.success:
            raise RuntimeError(f'Solver failed: {sol.message}')

        final = sol.y[:, -1]
        return {'V': final[0].item(), 'm': final[1].item(), 'h': final[2].item(), 'n': final[3].item()}