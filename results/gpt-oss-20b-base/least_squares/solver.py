import numpy as np
from scipy.optimize import leastsq
from numba import njit
from typing import Any, Callable, Tuple

# ---------------------------------------------------------------------------

@njit
def safe_exp(z: np.ndarray) -> np.ndarray:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

# ---------------------------------------------------------------------------

class Solver:
    # -----------------------------------------------------------------------
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fit model to data using Levenberg-Marquardt (leastsq)."""
        residual_fun, guess = self._create_residual_function(problem)

        # Fit parameters
        params_opt, cov_x, info, mesg, ier = leastsq(
            residual_fun, guess, full_output=True, maxfev=10000
        )

        # Prepare results
        x_arr = np.asarray(problem["x_data"])
        y_arr = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        if model_type == "polynomial":
            y_fit = np.polyval(params_opt, x_arr)
        elif model_type == "exponential":
            a, b, c = params_opt
            y_fit = a * safe_exp(b * x_arr) + c
        elif model_type == "logarithmic":
            a, b, c, d = params_opt
            y_fit = a * np.log(b * x_arr + c) + d
        elif model_type == "sigmoid":
            a, b, c, d = params_opt
            y_fit = a / (1 + safe_exp(-b * (x_arr - c))) + d
        else:  # sinusoidal
            a, b, c, d = params_opt
            y_fit = a * np.sin(b * x_arr + c) + d

        residuals = y_arr - y_fit
        residuals_sq = residuals * residuals
        mse = float(np.mean(residuals_sq))

        result = {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": ier in {1, 2, 3, 4},
                "status": int(ier),
                "message": mesg,
                "num_function_calls": int(info["nfev"]),
                "final_cost": float(np.sum(residuals_sq)),
            },
        }
        return result

    # -----------------------------------------------------------------------
    def _create_residual_function(
        self, problem: dict[str, Any]
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
        """Return a residual function suited for the chosen model and an initial guess."""
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        if model_type == "polynomial":
            deg = problem["degree"]

            @njit
            def resid(p: np.ndarray) -> np.ndarray:
                return y_data - np.polyval(p, x_data)

            guess = np.ones(deg + 1)

        elif model_type == "exponential":

            @njit
            def resid(p: np.ndarray) -> np.ndarray:
                a, b, c = p
                return y_data - (a * safe_exp(b * x_data) + c)

            guess = np.array([1.0, 0.05, 0.0])

        elif model_type == "logarithmic":

            @njit
            def resid(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)

            guess = np.array([1.0, 1.0, 1.0, 0.0])

        elif model_type == "sigmoid":

            @njit
            def resid(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y_data - (a / (1 + safe_exp(-b * (x_data - c))) + d)

            guess = np.array([3.0, 0.5, np.median(x_data), 0.0])

        elif model_type == "sinusoidal":

            @njit
            def resid(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)

            guess = np.array([2.0, 1.0, 0.0, 0.0])

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return resid, guess