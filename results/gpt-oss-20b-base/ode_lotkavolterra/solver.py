# solver.py

from __future__ import annotations

import numpy as np

class Solver:
    """Fast deterministic solver for Lotka‑Volterra equations."""

    # Pre‑compiled parameters are unpacked once per instance to avoid
    # dictionary lookups during the integration loop.
    def __init__(self) -> None:
        self._alpha: float | None = None
        self._beta: float | None = None
        self._delta: float | None = None
        self._gamma: float | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Integrate the ODE system and return the final state as a list.

        Parameters
        ----------
        problem: dict
            Contains initial state ``y0`` (array), time bounds ``t0`` and ``t1``,
            and the constant parameters in ``params`` dictionary.

        Returns
        -------
        dict[str, list[float]]
            Dictionary with a single key ``"solution"`` holding the final
            state values as Python floats.
        """
        self._extract_params(problem["params"])
        y0 = np.asarray(problem["y0"], dtype=np.float64, copy=False)
        t0, t1 = problem["t0"], problem["t1"]

        # Fixed‑step RK4 – 2000 samples gives good precision while keeping
        # the operation count trivial.  The step size can be tuned if an
        # accuracy target is known.
        N = 2000
        h = (t1 - t0) / N
        y = y0.copy()
        for _ in range(N):
            k1 = self._lotka_volterra(y)
            k2 = self._lotka_volterra(y + 0.5 * h * k1)
            k3 = self._lotka_volterra(y + 0.5 * h * k2)
            k4 = self._lotka_volterra(y + h * k3)
            y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        return {"solution": y.tolist()}

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _extract_params(self, params: dict[str, float]) -> None:
        """
        Store parametric constants in instance attributes for fast access.
        """
        self._alpha = params["alpha"]
        self._beta = params["beta"]
        self._delta = params["delta"]
        self._gamma = params["gamma"]

    def _lotka_volterra(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the RHS of the Lotka‑Volterra system.

        Parameters
        ----------
        state: np.ndarray
            Current state vector [x, y].

        Returns
        -------
        np.ndarray
            Derivative vector [dx/dt, dy/dt].
        """
        x, y = state[0], state[1]
        alpha, beta, delta, gamma = (
            self._alpha,
            self._beta,
            self._delta,
            self._gamma,
        )
        dx_dt = alpha * x - beta * x * y
        dy_dt = delta * x * y - gamma * y
        return np.array([dx_dt, dy_dt], dtype=np.float64)