# solver.py

from typing import Any, Dict, List, Union
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, types
from numba import prange

# Compile the Van der Pol RHS once.  The function signature is
#   (float, float[2]) -> float[2]
@njit
def _vdp(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    x, v = y[0], y[1]
    dx = v
    dv = mu * ((1.0 - x * x) * v - x)
    return np.array([dx, dv], dtype=np.float64)

class Solver:
    """
    Solver for the Van der Pol oscillator

    Parameters
    ----------
    None

    Notes
    -----
    * The rhs is JIT‑compiled with **numba** to avoid the cost of creating a new
      `np.array` on each call.
    * `solve_ivp` is called with very tight tolerances so that the dense
      output (which would otherwise return an array of size 1000) is avoided.
    * The final state is returned as a plain Python list of floats.
    """

    # Cache the compiled rhs so it is not compiled on every call
    _compiled_rhs = staticmethod(_vdp)

    def solve(self, problem: Dict[str, Union[np.ndarray, float]]) -> Dict[str, List[float]]:
        """
        Solve the ODE defined in *problem*.

        Parameters
        ----------
        problem : dict
            Keys:
            - 'y0' : initial state (array of length 2)
            - 't0' : start time (float)
            - 't1' : final time (float)
            - 'mu' : scalar parameter of the Van der Pol oscillator

        Returns
        -------
        dict
            Keys:
            - 'x' : final x value
            - 'v' : final v value
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            # convert to a Python list
            return {'x': float(sol.y[0, -1]), 'v': float(sol.y[1, -1])}
        else:
            raise RuntimeError(f'Solver failed: {sol.message}')

    def _solve(
        self,
        problem: Dict[str, Union[np.ndarray, float]],
        debug: bool = False,
    ) -> Any:
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])

        # The RHS needs the gauge parameter mu; we wrap it in a closure
        def rhs(t, y):
            return self._compiled_rhs(t, y, mu)

        rtol = 1e-08
        atol = 1e-09
        method = "Radau"

        # When debugging we want dense output and a fixed evaluation grid.
        t_eval = None
        if debug:
            t_eval = np.linspace(t0, t1, 1000)

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