# solver.py
from typing import Any, Callable
import numpy as np
from scipy.optimize import leastsq
import numba as nb

# ---------- fast, Numba‑accelerated helper functions ----------
@nb.njit
def _safe_exp_vec(z: np.ndarray) -> np.ndarray:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

@nb.njit
def _poly_residual(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Horner's method is fast and Numba–friendly
    val = 0.0
    for coeff in p[::-1]:
        val = val * x + coeff
    return y - val

@nb.njit
def _exp_residual(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b, c = p[0], p[1], p[2]
    return y - (a * _safe_exp_vec(b * x) + c)

@nb.njit
def _log_residual(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b, c, d = p[0], p[1], p[2], p[3]
    return y - (a * np.log(b * x + c) + d)

@nb.njit
def _sigmoid_residual(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b, c, d = p[0], p[1], p[2], p[3]
    inner = -b * (x - c)
    return y - (a / (1.0 + _safe_exp_vec(inner)) + d)

@nb.njit
def _sin_residual(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a, b, c, d = p[0], p[1], p[2], p[3]
    return y - (a * np.sin(b * x + c) + d)

# ---------- Solver implementation ----------
class Solver:
    """
    Very small, high‑performance curve‑fitting helper.

    The ``solve`` method accepts a dictionary describing the data and model
    and returns fitted parameters, residuals, the MSE and some convergence info.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        residual, guess = self._create_residual_function(problem)

        # Run the Levenberg–Marquardt routine
        params_opt, cov_x, info, mesg, ier = leastsq(
            residual, guess, full_output=True, maxfev=10_000
        )

        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        # Predict the fitted curve
        if model_type == "polynomial":
            y_fit = np.polyval(params_opt, x_data)
        elif model_type == "exponential":
            a, b, c = params_opt
            y_fit = a * _safe_exp_vec(b * x_data) + c
        elif model_type == "logarithmic":
            a, b, c, d = params_opt
            y_fit = a * np.log(b * x_data + c) + d
        elif model_type == "sigmoid":
            a, b, c, d = params_opt
            y_fit = a / (1 + _safe_exp_vec(-b * (x_data - c))) + d
        else:  # sinusoidal
            a, b, c, d = params_opt
            y_fit = a * np.sin(b * x_data + c) + d

        residuals = y_data - y_fit
        mse = float(np.mean(residuals**2))

        return {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": ier in {1, 2, 3, 4},
                "status": int(ier),
                "message": mesg,
                "num_function_calls": int(info["nfev"]),
                "final_cost": float(np.sum(residuals**2)),
            },
        }

    # ---------- Build the residual function ----------
    def _create_residual_function(
        self, problem: dict[str, Any]
    ) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        if model_type == "polynomial":
            deg = problem["degree"]

            def r(p: np.ndarray) -> np.ndarray:  # type: ignore
                return _poly_residual(p, x_data, y_data)

            guess = np.ones(deg + 1)

        elif model_type == "exponential":

            def r(p: np.ndarray) -> np.ndarray:
                return _exp_residual(p, x_data, y_data)

            guess = np.array([1.0, 0.05, 0.0], dtype=np.float64)

        elif model_type == "logarithmic":

            def r(p: np.ndarray) -> np.ndarray:
                return _log_residual(p, x_data, y_data)

            guess = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)

        elif model_type == "sigmoid":

            def r(p: np.ndarray) -> np.ndarray:
                return _sigmoid_residual(p, x_data, y_data)

            guess = np.array(
                [3.0, 0.5, np.median(x_data), 0.0], dtype=np.float64
            )

        elif model_type == "sinusoidal":

            def r(p: np.ndarray) -> np.ndarray:
                return _sin_residual(p, x_data, y_data)

            guess = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float64)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return r, guess