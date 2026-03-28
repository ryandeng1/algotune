from typing import Any, Dict, List
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

class Solver:
    # --------------------- public interface --------------------- #
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """
        Solve the Van der Pol oscillator.
        """
        sol = self._solve(problem, debug=False)
        if sol.success:
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")

    # --------------------- internal helpers --------------------- #
    @staticmethod
    @njit
    def _vdp(t: float, y: np.ndarray, mu: float) -> np.ndarray:
        """
        Van der Pol ODE system.  This JIT‑compiled routine is vastly faster
        than the Python equivalent used by scipy’s integrator.
        """
        x, v = y
        return np.array([v, mu * ((1 - x * x) * v - x)])

    def _solve(self, problem: Dict[str, np.ndarray | float], debug: bool = True) -> Any:
        """
        Delegate to scipy.integrate.solve_ivp using the accelerated ODE.
        """
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = float(problem["t0"]), float(problem["t1"])
        mu = float(problem["mu"])

        # SciPy expects a callable that accepts (t, y[, args])
        # We pass a small wrapper that forwards the mu argument.
        def rhs(t, y):
            return self._vdp(t, y, mu)

        # Dense output is useful when debugging but costs additional time.
        dense = debug
        t_eval = np.linspace(t0, t1, 1000, dtype=np.float64) if debug else None

        # Use a stiff solver that behaves well for large |mu|.
        sol = solve_ivp(
            rhs,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-8,
            atol=1e-9,
            t_eval=t_eval,
            dense_output=dense,
        )
        return sol