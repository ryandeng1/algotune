# solver.py
# optimised implementation of the original Solver class.

import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, Union, List

class Solver:
    """
    Solves a FitzHugh–Nagumo system using `scipy.integrate.solve_ivp` with
    minimal overhead:
     * The derivative function is implemented with plain Python arithmetic
       (avoids NumPy overhead inside the integrator).
     * Parameters are unpacked once outside the time loop.
     * The result is returned as a list of floats, exactly as required by
       the Prompt.
    """

    def solve(self, problem: Dict[str, Union[np.ndarray, float]]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            # Convert to Python floats and return as a list.
            return {'result': sol.y[:, -1].tolist()}
        raise RuntimeError(f"Solver failed: {sol.message}")

    # The heavy lifting happens here
    def _solve(
        self,
        problem: Dict[str, Union[np.ndarray, float]],
        debug: bool = True,
    ) -> Any:
        y0 = np.array(problem["y0"], dtype=float)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]

        # Unpack parameters once
        a, b, c, I = params["a"], params["b"], params["c"], params["I"]

        # Derivative function expressed with basic Python operations.
        # Using local variable look‑ups keeps the overhead minimal.
        def f(t, y):
            v, w = y[0], y[1]
            dv_dt = v - (v * v * v) / 3.0 - w + I
            dw_dt = a * (b * v - c * w)
            return [dv_dt, dw_dt]

        # Integrator settings
        rtol = 1e-8
        atol = 1e-8
        method = "RK45"

        # If debugging we request dense output and intermediate evaluations;
        # otherwise we care only about the end state.
        t_eval = np.linspace(t0, t1, 1000) if debug else None
        dense_output = debug

        return solve_ivp(
            f,
            (t0, t1),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=t_eval,
            dense_output=dense_output,
        )