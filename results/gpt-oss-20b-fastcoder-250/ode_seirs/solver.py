from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """
    Solver for the SEIRS epidemic model.
    Uses scipy.solve_ivp with a single call to compute the state at t1.
    """

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the SEIRS model and return the state at time t1.

        Parameters
        ----------
        problem : dict
            Keys:
                - "t0": float, initial time
                - "t1": float, final time
                - "y0": list[float] of length 4, initial compartment fractions
                - "params": dict with keys "beta", "sigma", "gamma", "omega"

        Returns
        -------
        list[float]
            The compartment fractions [S, E, I, R] at t1.
        """
        # Extract problem data
        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        params = problem["params"]
        beta = float(params["beta"])
        sigma = float(params["sigma"])
        gamma = float(params["gamma"])
        omega = float(params["omega"])

        # Define ODE system (local for speed)
        def seirs(t, y):
            S, E, I, R = y
            dS = -beta * S * I + omega * R
            dE = beta * S * I - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I - omega * R
            return np.array([dS, dE, dI, dR], dtype=float)

        # Solve ODE to the final time only
        sol = solve_ivp(
            seirs,
            (t0, t1),
            y0,
            method="RK45",      # same as reference
            rtol=1e-10,
            atol=1e-10,
            dense_output=False,  # no dense output
            t_eval=None,          # no intermediate times
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        # Extract final state
        return sol.y[:, -1].tolist()
