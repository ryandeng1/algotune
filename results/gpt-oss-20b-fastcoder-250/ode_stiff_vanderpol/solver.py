from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    """
    A lightweight high‑performance ODE solver that delegates to SciPy's
    :func:`scipy.integrate.solve_ivp`.  It is intentionally minimal to keep
    overhead low while still providing robust error handling.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve a boundary value problem defined by *problem* and return the
        final state as a list of floats.

        Parameters
        ----------
        problem : dict
            Must contain the keys:
            - ``func``: Callable ``f(t, y)`` returning ``dy/dt``.
            - ``y0``: Initial state array.
            - ``t_span``: Tuple ``(t_start, t_end)``.
            - ``t_eval``: ``ndarray`` of times where solution is required.
            Optional keyword arguments are forwarded to :func:`solve_ivp`.

        Returns
        -------
        dict
            Dictionary with a single key ``"final_state"`` whose value is the
            final state array converted to a list.
        """
        # Forward all keyword arguments except the fixed ones
        kwargs = {
            k: v
            for k, v in problem.items()
            if k not in {"func", "y0", "t_span", "t_eval"}
        }

        sol = solve_ivp(
            fun=problem["func"],
            t_span=problem["t_span"],
            y0=problem["y0"],
            t_eval=problem["t_eval"],
            **kwargs,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Return only the final state as a plain list
        return {"final_state": sol.y[:, -1].tolist()}