import numpy as np
from typing import Any, Callable, Dict, List, Tuple

# ----------------------------------------------------------------------
# Small, highly‑optimised ODE solver – hand‑written RK4 with a few tricklets
# ----------------------------------------------------------------------
def _rk4_step(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
              t: np.ndarray,
              y: np.ndarray,
              dt: float) -> np.ndarray:
    """Single RK4 step for vectorised y."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _solve_problem(problem: Dict[str, Any],
                   debug: bool = False) -> Tuple[np.ndarray, bool, str]:
    """
    Very small solver wrapper.
    `problem` must contain:
        * f     : callable giving derivative at (t, y)
        * y0    : initial state, 1‑D array
        * t     : 1‑D array of time points – expects constant spacing
    """
    f: Callable[[np.ndarray, np.ndarray], np.ndarray] = problem["f"]
    y0: np.ndarray = problem["y0"]
    t: np.ndarray = problem["t"]

    # Safety / sanity checks (costly but one‑shot)
    if not isinstance(t, np.ndarray) or t.ndim != 1:
        raise ValueError("t must be a 1‑D numpy array")
    if not isinstance(y0, np.ndarray) or y0.ndim != 1:
        raise ValueError("y0 must be a 1‑D numpy array")

    # Pre‑allocate output matrix: shape (len(y0), len(t))
    y = np.empty((y0.size, t.size), dtype=y0.dtype)
    y[:, 0] = y0

    # Fixed step size – derive dt once
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        # fallback to adaptive stepping for non‑uniform grid
        for i in range(1, t.size):
            y[:, i] = _rk4_step(f, np.array([t[i - 1]]), y[:, i - 1], t[i] - t[i - 1])
        return y, True, ""

    # Uniform grid – vectorised loop is fastest for small steps
    for i in range(1, t.size):
        y[:, i] = _rk4_step(f, np.array([t[i - 1]]), y[:, i - 1], dt)

    return y, True, ""


class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """Return the final state as a list of floats."""
        y, success, message = _solve_problem(problem, debug=False)

        if not success:
            raise RuntimeError(f"Solver failed: {message}")

        # convert to python list for the requested format
        final = y[:, -1].tolist()
        return {"solution": final}