from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, prange

# Pre‑compile the right‑hand side with Numba for speed
@njit
def _rhs(t, y, params):
    V, m, h, n = y
    C_m   = params["C_m"]
    g_Na  = params["g_Na"]
    g_K   = params["g_K"]
    g_L   = params["g_L"]
    E_Na  = params["E_Na"]
    E_K   = params["E_K"]
    E_L   = params["E_L"]
    I_app = params["I_app"]

    # avoid division by zero with a small epsilon
    tol = 1e-12
    V40 = V + 40.0
    V55 = V + 55.0
    V65 = V + 65.0
    V80 = V + 80.0

    # Rate constants
    alpha_m = 0.1 * V40 / (1.0 - np.exp(-V40/10.0 + tol))
    beta_m  = 4.0 * np.exp(-V65/18.0)
    alpha_h = 0.07 * np.exp(-V65/20.0)
    beta_h  = 1.0 / (1.0 + np.exp(-(V + 35.0)/10.0))
    alpha_n = 0.01 * V55 / (1.0 - np.exp(-V55/10.0 + tol))
    beta_n  = 0.125 * np.exp(-V65/80.0)

    # Clamp gating variables (Numba supports np.clip)
    m = np.clip(m, 0.0, 1.0)
    h = np.clip(h, 0.0, 1.0)
    n = np.clip(n, 0.0, 1.0)

    # Currents
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K  = g_K * n**4 * (V - E_K)
    I_L  = g_L * (V - E_L)

    dVdt = (I_app - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m * (1.0 - m) - beta_m * m
    dhdt = alpha_h * (1.0 - h) - beta_h * h
    dndt = alpha_n * (1.0 - n) - beta_n * n

    return np.array([dVdt, dmdt, dhdt, dndt])

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug=True) -> Any:
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Prepare Numba-compatible parameter dictionary
        numba_params = {
            "C_m": np.float64(params["C_m"]),
            "g_Na": np.float64(params["g_Na"]),
            "g_K": np.float64(params["g_K"]),
            "g_L": np.float64(params["g_L"]),
            "E_Na": np.float64(params["E_Na"]),
            "E_K": np.float64(params["E_K"]),
            "E_L": np.float64(params["E_L"]),
            "I_app": np.float64(params["I_app"]),
        }

        # Define a wrapper that calls the compiled RHS
        def rhs(t, y):
            return _rhs(t, y, numba_params)

        rtol = 1e-8
        atol = 1e-8
        method = "RK45"

        # If debugging, evaluate at many points to get dense output
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        sol = solve_ivp(
            rhs,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol