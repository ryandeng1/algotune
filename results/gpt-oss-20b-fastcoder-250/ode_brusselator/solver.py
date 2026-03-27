from __future__ import annotations
from typing import Dict, Any, List
import numpy as np


class Solver:
    """
    A small but efficient solver for linear time‑invariant ODEs

        dy/dt = A @ y + b

    The `problem` dictionary must contain:
        - "A"  : (n, n) numpy array, system matrix
        - "b"  : (n,)   numpy array, constant vector
        - "y0" : (n,)   numpy array, initial state
        - "t"  : (m,)   numpy array, time grid (monotonically increasing)

    The solver uses a 4th‑order Runge–Kutta method with adaptive step
    decision based on the local truncation error.  The implementation
    is fully vectorised with NumPy and avoids Python loops wherever
    possible for maximum speed.
    """

    # ---- public API -------------------------------------------------------
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        """
        Compute the solution of the ODE on the supplied time grid.

        Parameters
        ----------
        problem : dict
            Dictionary containing the problem data as explained in the class
            docstring.  All arrays are expected to be ``np.ndarray`` of
            type ``float64``.

        Returns
        -------
        result : dict
            Dictionary with the keys:
                "t"  : list[float]  times (identical to problem["t"])
                "y"  : list[float]  final state at the last time point
            The function raises `RuntimeError` if integration fails.
        """
        # Extract required data
        A = np.asarray(problem["A"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64).reshape(-1)
        y0 = np.asarray(problem["y0"], dtype=np.float64).reshape(-1)
        t = np.asarray(problem["t"], dtype=np.float64)

        # Quick sanity checks
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")
        if A.shape[0] != b.size:
            raise ValueError("Incompatible dimensions: A.shape[0] != b.size.")
        if y0.size != A.shape[0]:
            raise ValueError("Incompatible dimensions: y0.size != A.shape[0].")
        if t.ndim != 1 or not np.all(np.diff(t) > 0):
            raise ValueError("Time array must be 1‑D and strictly increasing.")

        # Run the integration ------------------------------------------------
        try:
            y_final = self._integrate(A, b, y0, t)
        except Exception as exc:
            raise RuntimeError(f"Integration failed: {exc}") from exc

        return {"t": t.tolist(), "y": y_final.tolist()}

    # ---- internal helpers -------------------------------------------------
    @staticmethod
    def _integrate(A: np.ndarray, b: np.ndarray, y0: np.ndarray,
                   t: np.ndarray) -> np.ndarray:
        """
        Integrate using a 4th order RK method with adaptive step size.
        The local error is estimated by comparing a full step with two half steps.
        """
        # Pre‑allocate state array and set initial state
        y = y0.copy()
        t_prev = t[0]

        # Constants for adaptive control
        safety = 0.9
        min_factor = 0.2
        max_factor = 5.0
        p = 4  # order of the method
        atol = 1e-9
        rtol = 1e-6

        # Helper for ODE RHS
        def f(t_now: float, y_now: np.ndarray) -> np.ndarray:
            return A @ y_now + b

        # Main loop over requested times
        for ti in t[1:]:
            h = ti - t_prev  # Desired step size for this interval

            # Simple but robust step‑acceptance logic
            for _ in range(10):
                # One full step
                k1 = f(t_prev, y)
                k2 = f(t_prev + 0.5 * h, y + 0.5 * h * k1)
                k3 = f(t_prev + 0.5 * h, y + 0.5 * h * k2)
                k4 = f(t_prev + h, y + h * k3)
                y_full = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                # Two half steps
                h2 = 0.5 * h
                # first half
                k1h = f(t_prev, y)
                k2h = f(t_prev + 0.5 * h2, y + 0.5 * h2 * k1h)
                k3h = f(t_prev + 0.5 * h2, y + 0.5 * h2 * k2h)
                k4h = f(t_prev + h2, y + h2 * k3h)
                y_half = y + (h2 / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)

                # second half
                k1h2 = f(t_prev + h2, y_half)
                k2h2 = f(t_prev + 0.5 * h2 + h2, y_half + 0.5 * h2 * k1h2)
                k3h2 = f(t_prev + 0.5 * h2 + h2, y_half + 0.5 * h2 * k2h2)
                k4h2 = f(t_prev + h, y_half + h2 * k3h2)
                y_half2 = y_half + (h2 / 6.0) * (k1h2 + 2 * k2h2 + 2 * k3h2 + k4h2)

                # Error estimate
                err = np.abs(y_full - y_half2)
                scale = atol + rtol * np.maximum(np.abs(y_full), np.abs(y_half2))
                err_norm = np.max(err / scale)

                if err_norm <= 1.0:
                    # Accept step
                    y = y_half2
                    t_prev = ti
                    # Adjust step size for next interval
                    if err_norm == 0:
                        fac = max_factor
                    else:
                        fac = min(max_factor,
                                  max(min_factor,
                                      safety * err_norm ** (-1.0 / (p + 1))))
                    h = fac * (ti - t_prev)  # Will be used if loop repeats
                    break
                else:
                    # Reject & reduce step size
                    fac = min(max_factor,
                              max(min_factor,
                                  safety * err_norm ** (-1.0 / (p + 1))))
                    h *= fac

            else:
                raise RuntimeError("Adaptive step size failed to converge.")

        return y