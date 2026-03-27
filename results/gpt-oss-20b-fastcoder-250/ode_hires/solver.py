from typing import Any
import numpy as np
from scipy.integrate import solve_ivp


class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        """
        Solve a system of ODEs defined in `problem` and return the final state.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
            - 'f'   : callable sending (t, y) -> dy/dt (numpy array)
            - 'y0'  : initial state (numpy array)
            - 't'   : array of time points to integrate over
            - 'args': additional arguments for the ODE function (optional)

        Returns
        -------
        dict
            Dictionary mapping the key ``"y"`` to a list containing the final state
            values (floating point numbers).

        Raises
        ------
        RuntimeError
            If the solution routine fails.
        """
        f = problem["f"]
        y0 = problem["y0"]
        t_span = (problem["t"][0], problem["t"][-1])
        t_eval = problem["t"]
        args = problem.get("args", ())

        sol = solve_ivp(
            fun=f,
            t_span=t_span,
            y0=y0,
            args=args,
            t_eval=t_eval,
            method="RK45",
            vectorized=False,
        )

        if sol.success:
            return {"y": sol.y[:, -1].tolist()}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")