import numpy as np
from typing import Any, Dict, List, Tuple, Callable


class Solver:
    """Fast vectorized ODE solver using a fixed RK4 scheme."""

    def __init__(self, dt: float = 1e-3, steps: int = 1000) -> None:
        self.dt = dt
        self.steps = steps

    def _ode_fun(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        User‑supplied ODE function.

        Expected signature: f(t, y) -> dydt.
        Override this method in subclasses or replace via monkey‑patching.
        """
        raise NotImplementedError("Define _ode_fun in subclass or monkey‑patch")

    def _solve(self, problem: Dict[str, Any]) -> Tuple[np.ndarray, bool, str]:
        """
        Core integration routine.

        Parameters
        ----------
        problem
            Dictionary containing:
            - `y0`: Initial state vector (ndarray).
            - `t0`: Initial time (float).
            - `t1`: Final time (float).
            - `ode`: Callable accepting (t, y) and returning dy/dt.
            - `dt`: Optional time step.  If omitted, class default is used.
            - `steps`: Optional number of integration steps.  If omitted,
                       class default is used.

        Returns
        -------
        y, success, message
            - `y`: 2‑D array of shape (n, nt) where each column is the state.
            - `success`: Boolean flag.
            - `message`: Error description if not successful.
        """
        # Extract integration parameters
        y0: np.ndarray = problem["y0"]
        t0: float = problem["t0"]
        t1: float = problem["t1"]
        ode: Callable[[float, np.ndarray], np.ndarray] = problem["ode"]
        dt = problem.get("dt", self.dt)
        steps = problem.get("steps", self.steps)

        # Pre‑allocate array for results
        n = y0.size
        y = np.zeros((n, steps + 1))
        y[:, 0] = y0
        t = t0

        # RK4 integration loop (vectorized computation of k1..k4)
        try:
            for i in range(steps):
                k1 = ode(t, y[:, i])
                k2 = ode(t + dt / 2.0, y[:, i] + dt * k1 / 2.0)
                k3 = ode(t + dt / 2.0, y[:, i] + dt * k2 / 2.0)
                k4 = ode(t + dt, y[:, i] + dt * k3)

                y[:, i + 1] = y[:, i] + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                t += dt
            return y, True, ""
        except Exception as exc:
            return np.array([]), False, str(exc)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Public API: solves the ODE defined in `problem`.

        Parameters
        ----------
        problem : dict
            Must contain keys listed in `_solve` docstring.

        Returns
        -------
        dict
            Mapping `"solution"` to a list of floats representing the final state.

        Raises
        ------
        RuntimeError
            If the integration fails.
        """
        y, success, msg = self._solve(problem)
        if success:
            return {"solution": y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {msg}")