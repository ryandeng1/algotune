import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List, Union

class Solver:
    """
    Optimised solution for the Lorenz‑96 system.
    """

    def solve(self, problem: Dict[str, Union[np.ndarray, float]]) -> Dict[str, List[float]]:
        """
        Only the final state at t1 is required.  solve_ivp is called with
        t_eval=None and dense_output=False for maximum speed.
        """
        sol = self._solve(problem, debug=False)

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return the final state as a Python list
        return sol.y[:, -1].tolist()

    # --------------------------------------------------------------------
    def _solve(self, problem: Dict[str, Union[np.ndarray, float]], debug: bool = True) -> Any:
        y0: np.ndarray = np.asarray(problem["y0"], dtype=float)
        t0: float = float(problem["t0"])
        t1: float = float(problem["t1"])
        F: float = float(problem["F"])

        # ----------------------------------------------------------------
        # Main vectorised Lorenz‑96 RHS
        def lorenz96(t: float, x: np.ndarray) -> np.ndarray:
            # Using in‑place operations to minimise memory traffic
            ip1 = np.roll(x, -1)      # x_{i+1}
            im1 = np.roll(x, 1)       # x_{i-1}
            im2 = np.roll(x, 2)       # x_{i-2}
            # (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
            return (ip1 - im2) * im1 - x + F

        # ----------------------------------------------------------------
        # Integrator settings – keep the defaults for speed
        rtol, atol = 1e-8, 1e-8
        method = "RK45"

        # Only evaluate at the final time, no intermediate points
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        sol = solve_ivp(
            lorenz96,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )

        return sol