import numpy as np
from numba import njit

@njit
def _alpha_m(V):
    if V == -40.0:
        return 1.0
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

@njit
def _beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

@njit
def _alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

@njit
def _beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

@njit
def _alpha_n(V):
    if V == -55.0:
        return 0.1
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

@njit
def _beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

@njit
def _nabla(V, m, h, n, parms):
    C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_app = parms
    m = np.clip(m, 0.0, 1.0)
    h = np.clip(h, 0.0, 1.0)
    n = np.clip(n, 0.0, 1.0)

    I_Na = g_Na * m ** 3 * h * (V - E_Na)
    I_K  = g_K * n ** 4 * (V - E_K)
    I_L  = g_L * (V - E_L)

    dVdt = (I_app - I_Na - I_K - I_L) / C_m
    dmdt = _alpha_m(V) * (1.0 - m) - _beta_m(V) * m
    dhdt = _alpha_h(V) * (1.0 - h) - _beta_h(V) * h
    dndt = _alpha_n(V) * (1.0 - n) - _beta_n(V) * n
    return dVdt, dmdt, dhdt, dndt

@njit
def _runge_kutta(step, t, y, parms):
    k1 = _nabla(t, y[0], y[1], y[2], y[3], parms)

    y2 = y + 0.5 * step * k1
    k2 = _nabla(t + 0.5 * step, y2[0], y2[1], y2[2], y2[3], parms)

    y3 = y + 0.5 * step * k2
    k3 = _nabla(t + 0.5 * step, y3[0], y3[1], y3[2], y3[3], parms)

    y4 = y + step * k3
    k4 = _nabla(t + step, y4[0], y4[1], y4[2], y4[3], parms)

    return y + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        y0 = np.array(problem['y0'], dtype=np.float64)
        t0, t1 = problem['t0'], problem['t1']
        parms = (
            problem['params']['C_m'],
            problem['params']['g_Na'],
            problem['params']['g_K'],
            problem['params']['g_L'],
            problem['params']['E_Na'],
            problem['params']['E_K'],
            problem['params']['E_L'],
            problem['params']['I_app'],
        )
        nsteps = 10000      # finer step for accuracy
        h = (t1 - t0) / nsteps
        y = y0.copy()
        for i in range(nsteps):
            t = t0 + i * h
            y = _runge_kutta(h, t, y, parms)
        return y.tolist()