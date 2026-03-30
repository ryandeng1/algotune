# solver.py
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

# ------------- fast ODE function ---------------------------------------
# We compile the RK function with Numba for speed.  It returns a
# numerically stable, tiny‐array (np.ndarray(3)) without allocating in a
# Python loop.
# NOTE: `rober` must be a Python function; we *decorate* it, otherwise
# SciPy would see a plain callable and warn.  Using `numba.njit` keeps
# the implementation on the C side and avoids the overhead of Python
# attribute look‑ups inside the integrator.
from numba import njit

@njit
def _rober_jit(t: float, y: np.ndarray, k: Tuple[float, float, float]) -> np.ndarray:
    # unpack y
    y1, y2, y3 = y
    k1, k2, k3 = k

    # compute derivatives
    f0 = -k1 * y1 + k3 * y2 * y3
    f1 =  k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3
    f2 =  k2 * y2 * y2

    out = np.empty(3, dtype=y.dtype)
    out[0] = f0
    out[1] = f1
    out[2] = f2
    return out

# -----------------------------------------------------------------------

class Solver:
    """Solver for the classic ROBCO ODE system.

    The implementation uses a Numba JIT compiled derivative function to
    avoid Python overhead in the nested loop of the IVP integrator.
    """

    def solve(self, problem: Dict[str, Union[np.ndarray, float]]) -> Dict[str, List[float]]:
        """
        Solve the ODE defined by the input dictionary `problem`.

        Parameters
        ----------
        problem : dict
            Must contain keys ``y0`` (array of initial concentrations), ``t0`` and
            ``t1`` (times), and ``k`` (list/tuple of rate constants).

        Returns
        -------
        dict
            Mapping each state variable to a list of its final values.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            # `y` is a 2‑D array, shape (n_vars, n_times)
            return sol.y[:, -1].tolist()
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")

    # -------------------------------------------------------------------
    def _solve(self, problem: Dict[str, Union[np.ndarray, float]], debug: bool = True) -> Any:
        """
        Internal solver that keeps all modalities (time grid, tolerances) and
        returns the raw integrator output.
        """
        y0 = np.asarray(problem["y0"], dtype=np.float64)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        k = tuple(float(v) for v in problem["k"])  # enforce tuple of floats

        # Callback for the integrator that forwards *k* as a constant
        def rober(t, y):
            return _rober_jit(t, y, k)

        # Integration settings
        rtol = 1e-11
        atol = 1e-9
        method = "Radau"

        # Build evaluation grid if desired
        if debug:
            # 1000 points distributed on a log‑scale to capture the early fast transients
            t_eval = np.clip(
                np.exp(np.linspace(np.log(1e-06), np.log(t1), 1000)),
                t0,
                t1,
            )
        else:
            t_eval = None

        # Run the integration
        sol = solve_ivp(
            rober,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=debug,
        )
        return sol