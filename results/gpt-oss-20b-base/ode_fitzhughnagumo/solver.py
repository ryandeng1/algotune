import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the FitzHugh-Nagumo ODE system for the final time t1.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
            - "t0": float, initial time
            - "t1": float, final time
            - "y0": list[float], initial state [v0, w0]
            - "params": dict with keys "a", "b", "c", "I"

        Returns
        -------
        List[float]
            State [v, w] at time t1.
        """
        y0 = np.array(problem["y0"], dtype=float)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        p = problem["params"]
        a, b, c, I = float(p["a"]), float(p["b"]), float(p["c"]), float(p["I"])

        # Vectorized derivative function
        def f(t, y):
            v, w = y
            dv = v - (v ** 3) / 3.0 - w + I
            dw = a * (b * v - c * w)
            return np.array([dv, dw], dtype=float)

        # Integrate once to final time
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            events=None,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        return sol.y[:, -1].tolist()
