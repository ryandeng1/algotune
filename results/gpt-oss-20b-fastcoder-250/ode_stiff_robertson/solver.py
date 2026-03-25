from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def __init__(self) -> None:
        # Pre‑compile the right‑hand side symbols for slightly faster calls
        self._k: tuple[float, float, float] | None = None

    def _rober(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right–hand side of the Robertson equations."""
        y1, y2, y3 = y
        k1, k2, k3 = self._k  # type: ignore[assignment]
        return np.array(
            [
                -k1 * y1 + k3 * y2 * y3,
                k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
                k2 * y2 * y2,
            ],
            dtype=np.float64,
        )

    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """Return the concentrations at t1 for the Robertson problem."""
        t0: float = float(problem["t0"])
        t1: float = float(problem["t1"])
        y0: np.ndarray = np.asarray(problem["y0"], dtype=np.float64)
        self._k = tuple(map(float, problem["k"]))

        # Use an implicit stiff solver (Radau or BDF) with tight tolerances
        sol = solve_ivp(
            self._rober,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-9,
            atol=1e-12,
            dense_output=False,
            max_step=t1 - t0,  # single step is allowed, solver will adapt internally
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()
