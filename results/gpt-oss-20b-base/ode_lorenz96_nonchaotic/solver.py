from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        # Extract problem parameters
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        F = float(problem["F"])
        N = y0.size

        # Lorenz‑96 RHS
        def lorenz96(t: float, x: np.ndarray) -> np.ndarray:
            # x[i-1] * (x[i+1] - x[i-2]) - x[i] + F
            return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

        # Solve ODE only for final time
        sol = solve_ivp(
            lorenz96,
            (t0, t1),
            y0,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
            dense_output=False,
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return final state as list
        return sol.y[:, -1].tolist()
