# solver.py
from __future__ import annotations
import numpy as np
from scipy.optimize import leastsq
from numba import njit

# --------------------------------------------------------------------------- #
# Helper functions – compiled with *numba* to eliminate Python overhead
# --------------------------------------------------------------------------- #
@njit
def _safe_exp_numba(z: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(z, -50.0, 50.0))

@njit
def _resid_poly(p: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    # p[0] is highest degree coefficient
    return y - np.polyval(p, x)

@njit
def _resid_exp(p: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, c = p[0], p[1], p[2]
    return y - (a * _safe_exp_numba(b * x) + c)

@njit
def _resid_log(p: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, c, d = p[0], p[1], p[2], p[3]
    return y - (a * np.log(b * x + c) + d)

@njit
def _resid_sigmoid(p: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, c, d = p[0], p[1], p[2], p[3]
    return y - (a / (1 + _safe_exp_numba(-b * (x - c))) + d)

@njit
def _resid_sinusoidal(p: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    a, b, c, d = p[0], p[1], p[2], p[3]
    return y - (a * np.sin(b * x + c) + d)


# --------------------------------------------------------------------------- #
# Solver class – thin wrapper around the fast residual functions
# --------------------------------------------------------------------------- #
class Solver:
    """
    Optimiser for a handful of popular 1‑D regression models.
    Uses the fast *numba* compiled residual functions together with
    SciPy's Levenberg–Marquardt implementation via ``leastsq``.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Input data
        x = np.asarray(problem["x_data"], dtype=np.float64)
        y = np.asarray(problem["y_data"], dtype=np.float64)
        mtype = problem["model_type"]

        # Choose the appropriate residual function and initial guess
        if mtype == "polynomial":
            deg = problem["degree"]
            r = lambda p: _resid_poly(p, y, x)
            guess = np.ones(deg + 1, dtype=np.float64)
        elif mtype == "exponential":
            r = lambda p: _resid_exp(p, y, x)
            guess = np.array([1.0, 0.05, 0.0], dtype=np.float64)
        elif mtype == "logarithmic":
            r = lambda p: _resid_log(p, y, x)
            guess = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
        elif mtype == "sigmoid":
            r = lambda p: _resid_sigmoid(p, y, x)
            guess = np.array([3.0, 0.5, np.median(x), 0.0], dtype=np.float64)
        elif mtype == "sinusoidal":
            r = lambda p: _resid_sinusoidal(p, y, x)
            guess = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float64)
        else:
            raise ValueError(f"Unknown model type: {mtype}")

        # Run the optimiser
        params_opt, cov_x, info, mesg, ier = leastsq(
            r,
            guess,
            full_output=True,
            maxfev=10_000,
        )

        # Construct fitted model
        if mtype == "polynomial":
            y_fit = np.polyval(params_opt, x)
        elif mtype == "exponential":
            a, b, c = params_opt
            y_fit = a * _safe_exp_numba(b * x) + c
        elif mtype == "logarithmic":
            a, b, c, d = params_opt
            y_fit = a * np.log(b * x + c) + d
        elif mtype == "sigmoid":
            a, b, c, d = params_opt
            y_fit = a / (1 + _safe_exp_numba(-b * (x - c))) + d
        else:  # sinusoidal
            a, b, c, d = params_opt
            y_fit = a * np.sin(b * x + c) + d

        residuals = y - y_fit
        mse = float(np.mean(residuals ** 2))

        return {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": int(ier in {1, 2, 3, 4}),
                "status": int(ier),
                "message": mesg,
                "num_function_calls": int(info["nfev"]),
                "final_cost": float(np.sum(residuals ** 2)),
            },
        }