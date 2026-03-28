import numpy as np

class Solver:
    """
    Fast explicit RK4 implementation of the Hodgkin–Huxley ODE.
    """
    def solve(self, problem: dict[str, np.ndarray | float]) -> list[float]:
        # initial state and parameters
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # number of steps (fixed to 1000 as in the original code)
        n_steps = 1000
        h = (t1 - t0) / n_steps

        # pre‑extract scalar parameters to avoid dictionary lookups
        C_m   = params["C_m"]
        g_Na  = params["g_Na"]
        g_K   = params["g_K"]
        g_L   = params["g_L"]
        E_Na  = params["E_Na"]
        E_K   = params["E_K"]
        E_L   = params["E_L"]
        I_app = params["I_app"]

        # local function for the RHS of the ODE
        def rhs(V, m, h, n):
            # alpha and beta values with analytic continuous forms
            # to avoid the branch on V == -40 and V == -55; use safe division
            exp1 = np.exp(-(V + 40.0) / 10.0)
            exp2 = np.exp(-(V + 55.0) / 10.0)
            alpha_m = (V + 40.0) * 0.1 / (1.0 - exp1)
            beta_m  = 4.0 * np.exp(-(V + 65.0) / 18.0)
            alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
            beta_h  = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            alpha_n = (V + 55.0) * 0.01 / (1.0 - exp2)
            beta_n  = 0.125 * np.exp(-(V + 65.0) / 80.0)

            # clamp gating variables
            m = np.clip(m, 0.0, 1.0)
            h = np.clip(h, 0.0, 1.0)
            n = np.clip(n, 0.0, 1.0)

            I_Na = g_Na * m**3 * h * (V - E_Na)
            I_K  = g_K  * n**4 * (V - E_K)
            I_L  = g_L  * (V - E_L)
            dVdt = (I_app - I_Na - I_K - I_L) / C_m

            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n
            return dVdt, dmdt, dhdt, dndt

        # initialise state vector
        V, m, h, n = y0
        for _ in range(n_steps):
            k1 = np.array(rhs(V, m, h, n), dtype=np.float64)

            V2, m2, h2, n2 = V + 0.5 * h * k1[0], m + 0.5 * h * k1[1], h + 0.5 * h * k1[2], n + 0.5 * h * k1[3]
            k2 = np.array(rhs(V2, m2, h2, n2), dtype=np.float64)

            V3, m3, h3, n3 = V + 0.5 * h * k2[0], m + 0.5 * h * k2[1], h + 0.5 * h * k2[2], n + 0.5 * h * k2[3]
            k3 = np.array(rhs(V3, m3, h3, n3), dtype=np.float64)

            V4, m4, h4, n4 = V + h * k3[0], m + h * k3[1], h + h * k3[2], n + h * k3[3]
            k4 = np.array(rhs(V4, m4, h4, n4), dtype=np.float64)

            V += h * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
            m += h * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
            h += h * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6.0
            n += h * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6.0

        # final state as list
        return [float(V), float(m), float(h), float(n)]