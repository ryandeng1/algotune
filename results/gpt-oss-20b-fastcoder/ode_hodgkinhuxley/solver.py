from typing import Any, Dict
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# ----------------------------------------------------------------------------- #
# pre‑compute ODE evaluator with numba for speed
# ----------------------------------------------------------------------------- #
@njit
def _hodgkin_huxley(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    NumPy/Numba implementation of the Hodgkin–Huxley model.
    The `params` array order: [C_m,g_Na,g_K,g_L,E_Na,E_K,E_L,I_app]
    """
    V, m, h, n = y
    # constants unpacked once
    C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app = params

    # rate functions with the usual analytical simplifications
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

    # keep gating variables in [0,1]
    # (they are already clipped in practice, but enforce to avoid numerical drift)
    if m > 1.0:
        m = 1.0
    elif m < 0.0:
        m = 0.0
    if h > 1.0:
        h = 1.0
    elif h < 0.0:
        h = 0.0
    if n > 1.0:
        n = 1.0
    elif n < 0.0:
        n = 0.0

    # currents
    I_Na = g_Na * m ** 3 * h * (V - E_Na)
    I_K = g_K * n ** 4 * (V - E_K)
    I_L = g_L * (V - E_L)

    # RHS
    dVdt = (I_app - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m * (1.0 - m) - beta_m * m
    dhdt = alpha_h * (1.0 - h) - beta_h * h
    dndt = alpha_n * (1.0 - n) - beta_n * n

    return np.array([dVdt, dmdt, dhdt, dndt])


# ----------------------------------------------------------------------------- #
# Solver class used by the benchmark harness
# ----------------------------------------------------------------------------- #
class Solver:
    """
    Fast Hodgkin–Huxley solver.

    The heavy lifting (the ODE kernel) is compiled with Numba.
    The outer wrapper only builds the constant array once per call.
    """

    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    # ------------------------------------------------------------------------- #
    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = False) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]

        # Pack all right‑hand side parameters into a dense NumPy array
        # this removes any per‑step dictionary look‑ups.
        p = np.empty(8, dtype=np.float64)
        p[0] = problem["params"]["C_m"]
        p[1] = problem["params"]["g_Na"]
        p[2] = problem["params"]["g_K"]
        p[3] = problem["params"]["g_L"]
        p[4] = problem["params"]["E_Na"]
        p[5] = problem["params"]["E_K"]
        p[6] = problem["params"]["E_L"]
        p[7] = problem["params"]["I_app"]

        # SciPy solve_ivp accepts a function of the form f(t, y)
        # We wrap our Numba routine to inject the constant parameter array.
        def _ode(t, y):
            return _hodgkin_huxley(t, y, p)

        # Select an appropriate solver:
        #   * RK45 works well for non‑stiff problems.
        #   * Dense output is disabled when not collecting intermediate points.
        method = "RK45"
        rtol, atol = 1e-8, 1e-8

        t_eval = (
            np.linspace(t0, t1, 1000, dtype=np.float64) if debug else None
        )
        dense_output = debug

        return solve_ivp(
            _ode,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=dense_output,
        )