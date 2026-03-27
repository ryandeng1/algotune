from typing import Dict, List, Union
import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem: Dict[str, Union[np.ndarray, float]]) -> Dict[str, List[float]]:
        """
        Solve an ODE system defined by the input problem dictionary.

        Expected keys in `problem`:
          - "fun": Callable representing the ODE system.
          - "t_span": Tuple or array-like with start and end times.
          - "y0": Initial state as a NumPy array.
          - Optional kwargs for solve_ivp: e.g. "t_eval", "method", etc.

        Returns a dictionary with the final state under the key "state".
        """
        # Unpack required arguments
        fun = problem["fun"]
        t_span = problem["t_span"]
        y0 = np.asarray(problem["y0"])

        # Pass all other optional parameters to solve_ivp
        ivp_kwargs = {k: v for k, v in problem.items()
                      if k not in {"fun", "t_span", "y0"}}

        # Solve the ODE using solve_ivp with a fast default method
        sol = solve_ivp(fun, t_span, y0, **ivp_kwargs)

        if sol.success:
            return {"state": sol.y[:, -1].tolist()}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")