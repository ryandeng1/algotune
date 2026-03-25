import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        t0 = problem["t0"]
        t1 = problem["t1"]
        y0 = np.array(problem["y0"])
        params = problem["params"]
        
        S0, E0, I0, R0 = y0
        y0_three = np.array([S0, E0, I0])
        
        def seirs(t, y):
            S, E, I = y
            R = 1.0 - S - E - I
            beta = params["beta"]
            sigma = params["sigma"]
            gamma = params["gamma"]
            omega = params["omega"]
            dSdt = -beta * S * I + omega * R
            dEdt = beta * S * I - sigma * E
            dIdt = sigma * E - gamma * I
            return np.array([dSdt, dEdt, dIdt])
        
        sol = solve_ivp(
            seirs,
            [t0, t1],
            y0_three,
            method="LSODA",
            rtol=1e-10,
            atol=1e-10,
            dense_output=False
        )
        
        S_final, E_final, I_final = sol.y[:, -1]
        R_final = 1.0 - S_final - E_final - I_final
        
        return [S_final, E_final, I_final, R_final]
