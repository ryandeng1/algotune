# solver.py
from typing import Any, List, Dict
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    """
    Fast Hodgkin–Huxley solver using SciPy's solve_ivp.
    """
    # Pre-allocate arrays for speed
    _alpha_m_cache: np.ndarray | None = None
    _alpha_n_cache: np.ndarray | None = None

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        y0 = np.array(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        params = problem["params"]

        C_m   = params["C_m"];   g_Na  = params["g_Na"];   g_K   = params["g_K"]
        g_L   = params["g_L"];   E_Na  = params["E_Na"];  E_K   = params["E_K"]
        E_L   = params["E_L"];   I_app = params["I_app"]

        # Define the RHS as a local function for speed
        def rhs(t, y):
            V, m, h, n = y

            # Alpha/Beta functions (avoid division by zero)
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

            # Clamp gating variables to [0, 1]
            m = m if 0.0 <= m <= 1.0 else np.clip(m, 0.0, 1.0)
            h = h if 0.0 <= h <= 1.0 else np.clip(h, 0.0, 1.0)
            n = n if 0.0 <= n <= 1.0 else np.clip(n, 0.0, 1.0)

            I_Na = g_Na * m**3 * h * (V - E_Na)
            I_K  = g_K  * n**4        * (V - E_K)
            I_L  = g_L  * (V - E_L)

            dVdt = (I_app - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n

            return np.array([dVdt, dmdt, dhdt, dndt], dtype=float)

        # Solve from t0 to t1 without intermediate evaluations
        sol = solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,   # we need only final state
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.y[:, -1].tolist()
