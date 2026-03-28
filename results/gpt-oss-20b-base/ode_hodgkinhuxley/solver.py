import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, list[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return [float(v) for v in sol.y[:, -1]]
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(self, problem: Dict[str, Any], debug: bool) -> Any:
        y0 = np.asarray(problem['y0'], dtype=float)
        t0, t1 = problem['t0'], problem['t1']
        params = problem['params']

        # extract parameters once
        C_m = params['C_m']
        g_Na = params['g_Na']
        g_K = params['g_K']
        g_L = params['g_L']
        E_Na = params['E_Na']
        E_K = params['E_K']
        E_L = params['E_L']
        I_app = params['I_app']

        N_STEPS = 1000 if debug else 1000  # fixed-step explicit RK4
        h = (t1 - t0) / N_STEPS

        def f(V, m, h_, n):
            # compute rates
            if abs(V + 40.0) < 1e-12:
                alpha_m = 1.0
            else:
                alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
            beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

            if abs(V + 55.0) < 1e-12:
                alpha_n = 0.1
            else:
                alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
            beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

            m = np.clip(m, 0.0, 1.0)
            h_ = np.clip(h_, 0.0, 1.0)
            n = np.clip(n, 0.0, 1.0)

            I_Na = g_Na * m ** 3 * h_ * (V - E_Na)
            I_K = g_K * n ** 4 * (V - E_K)
            I_L = g_L * (V - E_L)
            dVdt = (I_app - I_Na - I_K - I_L) / C_m

            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h_) - beta_h * h_
            dndt = alpha_n * (1.0 - n) - beta_n * n
            return dVdt, dmdt, dhdt, dndt

        V, m, h_, n = y0
        for _ in range(N_STEPS):
            k1 = f(V, m, h_, n)
            k2 = f(V + 0.5 * h * k1[0], m + 0.5 * h * k1[1],
                     h_ + 0.5 * h * k1[2], n + 0.5 * h * k1[3])
            k3 = f(V + 0.5 * h * k2[0], m + 0.5 * h * k2[1],
                     h_ + 0.5 * h * k2[2], n + 0.5 * h * k2[3])
            k4 = f(V + h * k3[0], m + h * k3[1],
                     h_ + h * k3[2], n + h * k3[3])

            V += (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            m += (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            h_ += (h / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            n += (h / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        class Result:
            success = True
            y = np.array([V, m, h_, n])[:, None]
        return Result()