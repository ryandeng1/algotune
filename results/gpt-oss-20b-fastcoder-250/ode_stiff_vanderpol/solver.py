from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the stiff Van der Pol oscillator and return the state at the final time.

        Parameters
        ----------
        problem : dict
            Dictionary containing the keys:
            - "mu": float, stiffness parameter.
            - "y0": list or array of two floats, initial state [x(0), v(0)].
            - "t0": float, initial time.
            - "t1": float, final time.

        Returns
        -------
        list[float]
            The state [x(t1), v(t1)].
        """
        # Parse problem dict
        mu = float(problem["mu"])
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])

        # Define RHS of ODE
        def vdp(t, y):
            x, v = y
            return np.array([v,
                             mu * ((1 - x * x) * v - x)])

        # Solve using a stiff solver
        sol = solve_ivp(
            vdp,
            [t0, t1],
            y0,
            method="Radau",
            rtol=1e-8,
            atol=1e-9,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # Return final state
        return sol.y[:, -1].tolist()
