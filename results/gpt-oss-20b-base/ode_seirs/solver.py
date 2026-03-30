from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from functools import partial
from typing import Any, Dict, List, Union

# ------------------------------------------------------------------
# NumPy-aware JIT‑compiled RHS for the SEIRS ODE system
# ------------------------------------------------------------------
@njit
def _seirs_rhs(
    _: float,                # time – unused but required by `solve_ivp`
    y: np.ndarray,           # state vector [S, E, I, R]
    beta: float,
    sigma: float,
    gamma: float,
    omega: float,
) -> np.ndarray:
    """
    Compute the derivative vector for the SEIRS model.
    The function is compiled with Numba for very fast evaluation
    during the ODE integration.
    """
    S, E, I, R = y
    dSdt = -beta * S * I + omega * R
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    return np.array([dSdt, dEdt, dIdt, dRdt], dtype=np.float64)


# ------------------------------------------------------------------
# Main solver class
# ------------------------------------------------------------------
class Solver:
    """
    Fast solver for a SEIRS epidemic model.
    The solver returns the final state as a list of four floats.
    """

    def solve(
        self, problem: Dict[str, Union[np.ndarray, float]]
    ) -> Dict[str, List[float]]:
        """
        Execute the ODE integration and return the final state.
        Raises ``RuntimeError`` if the integration fails.
        """
        result = self._solve(problem, debug=False)
        if result.success:
            # Convert to plain Python list for the public API
            return result.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {result.message}")

    def _solve(
        self, problem: Dict[str, Union[np.ndarray, float]], debug: bool = True
    ) -> Any:
        """
        Internal helper that sets up the integration.
        The ``debug`` option is kept for compatibility but defaults to False
        in the public ``solve`` call.
        """
        # Extract initial state and time bounds
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        beta, sigma, gamma, omega = (
            params["beta"],
            params["sigma"],
            params["gamma"],
            params["omega"],
        )

        # Wrap the JIT‐compiled RHS so that it matches the signature expected by solve_ivp
        rhs = partial(
            _seirs_rhs, beta=beta, sigma=sigma, gamma=gamma, omega=omega
        )

        # Solver settings – small tolerances for high accuracy
        rtol = 1e-10
        atol = 1e-10
        method = "RK45"

        # Evaluate at many points only if debugging
        t_eval = np.linspace(t0, t1, 1000) if debug else None

        # Run the integration
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol