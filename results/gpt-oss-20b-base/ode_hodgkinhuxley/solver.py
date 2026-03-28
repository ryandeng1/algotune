from typing import Any
import numpy as np
from math import exp, isclose
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    def _solve(self, problem: dict[str, np.ndarray | float], debug: bool = False) -> Any:
        y0 = np.array(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        p = problem["params"]

        C_m = float(p["C_m"])
        g_Na = float(p["g_Na"])
        g_K = float(p["g_K"])
        g_L = float(p["g_L"])
        E_Na = float(p["E_Na"])
        E_K = float(p["E_K"])
        E_L = float(p["E_L"])
        I_app = float(p["I_app"])

        def hodgkin_huxley(t: float, y: np.ndarray) -> np.ndarray:
            V, m, h, n = y
            if isclose(V, -40.0):
                alpha_m = 1.0
            else:
                alpha_m = 0.1 * (V + 40.0) / (1.0 - exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * exp(-(V + 65.0) / 18.0)

            alpha_h = 0.07 * exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))

            if isclose(V, -55.0):
                alpha_n = 0.1
            else:
                alpha_n = 0.01 * (V + 55.0) / (1.0 - exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * exp(-(V + 65.0) / 80.0)

            m = min(max(m, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            n = min(max(n, 0.0), 1.0)

            I_Na = g_Na * (m ** 3) * h * (V - E_Na)
            I_K = g_K * (n ** 4) * (V - E_K)
            I_L = g_L * (V - E_L)

            dVdt = (I_app - I_Na - I_K - I_L) / C_m
            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n

            return np.array([dVdt, dmdt, dhdt, dndt], dtype=float)

        t_eval = np.linspace(t0, t1, 1000) if debug else None
        sol = solve_ivp(
            hodgkin_huxley,
            [t0, t1],
            y0,
            method="RK45",
            t_eval=t_eval,
            dense_output=debug,
            rtol=1e-8,
            atol=1e-8,
        )
        return sol