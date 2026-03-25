# solver.py
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """Synchronous Hodgkin–Huxley neuron solver."""

    @staticmethod
    def _alpha_beta(V: float) -> tuple[float, float, float, float, float, float]:
        """Compute all alpha/beta rate constants for a given membrane voltage."""
        # alpha_m
        if V == -40.0:
            alpha_m = 1.0
        else:
            alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

        alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

        # alpha_n
        if V == -55.0:
            alpha_n = 0.1
        else:
            alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

    @staticmethod
    def _equations(t: float, y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Right‑hand side of the Hodgkin–Huxley ODE system."""
        V, m, h, n = y

        # Parameters
        C_m = params["C_m"]
        g_Na = params["g_Na"]
        g_K = params["g_K"]
        g_L = params["g_L"]
        E_Na = params["E_Na"]
        E_K = params["E_K"]
        E_L = params["E_L"]
        I_app = params["I_app"]

        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = Solver._alpha_beta(V)

        # Clip gating variables to [0,1] for numerical safety
        m = np.clip(m, 0.0, 1.0)
        h = np.clip(h, 0.0, 1.0)
        n = np.clip(n, 0.0, 1.0)

        # Ionic currents
        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_K = g_K * n**4 * (V - E_K)
        I_L = g_L * (V - E_L)

        dVdt = (I_app - I_Na - I_K - I_L) / C_m
        dmdt = alpha_m * (1.0 - m) - beta_m * m
        dhdt = alpha_h * (1.0 - h) - beta_h * h
        dndt = alpha_n * (1.0 - n) - beta_n * n

        return np.array([dVdt, dmdt, dhdt, dndt], dtype=np.float64)

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """Integrate the Hodgkin–Huxley model from t0 to t1."""
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Use a reasonably efficient solver with minimal step control
        sol = solve_ivp(
            fun=lambda t, y: self._equations(t, y, params),
            t_span=(t0, t1),
            y0=y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.y[:, -1].tolist()
