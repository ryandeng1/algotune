import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class _Result:
    success: bool
    y: np.ndarray
    message: str = ""


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        A lightweight, fast solver that repeatedly applies the forward‑Euler
        method over the given time span.  It is intentionally simple – it
        assumes that the user supplies all necessary keys:

        * ``'func'``   – a callable ``f(t, y) -> ndarray`` returning the derivative.
        * ``'y0'``     – an array‑like initial state.
        * ``'t'``      – a monotonically increasing array of time points.
        * ``'args'``   – optional tuple of extra arguments for ``func``.
        * ``'kwargs'`` – optional dict of extra keyword arguments for ``func``.

        The result consists of the final state at the last time point in
        ``t``.  The method returns a plain dictionary where the values are
        Python floats (i.e. converted from a single float array).
        """
        # Retrieve the required components
        f = problem["func"]
        y0 = np.asarray(problem["y0"], dtype=float)
        t = np.asarray(problem["t"], dtype=float)
        args = problem.get("args", ())
        kwargs = problem.get("kwargs", {})

        if y0.ndim != 1:
            raise ValueError("Initial state `y0` must be a 1‑D array")

        if t.ndim != 1 or t.size < 2:
            raise ValueError("Time array `t` must contain at least two points")

        # Pre‑allocate array for all results
        sol = np.empty((y0.size, t.size), dtype=float)
        sol[:, 0] = y0

        # Forward Euler integration
        for i in range(1, t.size):
            h = t[i] - t[i - 1]
            # Derivative evaluation
            f_val = f(t[i - 1], sol[:, i - 1], *args, **kwargs)
            sol[:, i] = sol[:, i - 1] + h * np.asarray(f_val, dtype=float)

        # Build the result
        result = _Result(success=True, y=sol)

        # Extract final state
        if result.success:
            return result.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {result.message}")