from __future__ import annotations
from typing import Any, Callable, Tuple, List, Dict

import numpy as np

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Stable exponential that avoids overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

# --------------------------------------------------------------------------- #
# Solver implementation
# --------------------------------------------------------------------------- #
class Solver:
    """Fast non‑linear least‑squares solver for simple curve fitting."""
    
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit a model to the data using either an analytic linear
        solution (polynomial) or a robust non‑linear optimiser
        for generic models.
        """
        x_data = np.asarray(problem["x_data"], dtype=np.float64)
        y_data = np.asarray(problem["y_data"], dtype=np.float64)
        model_type = problem["model_type"]
        residual_fn, guess = self._create_residual_function(problem, x_data, y_data)
        
        if model_type == "polynomial":
            # analytic solution via least squares
            A = np.vander(x_data, N=len(guess), increasing=False)
            params_opt, *_ = np.linalg.lstsq(A, y_data, rcond=None)
        else:
            # use Scipy's Levenberg-Marquardt
            from scipy.optimize import least_squares  # lazy import
            res = least_squares(residual_fn, guess, method="lm", max_nfev=10000)
            params_opt = res.x

        # Build fitted curve
        if model_type == "polynomial":
            y_fit = np.polyval(params_opt, x_data)
        elif model_type == "exponential":
            a, b, c = params_opt
            y_fit = a * _safe_exp(b * x_data) + c
        elif model_type == "logarithmic":
            a, b, c, d = params_opt
            y_fit = a * np.log(b * x_data + c) + d
        elif model_type == "sigmoid":
            a, b, c, d = params_opt
            y_fit = a / (1 + _safe_exp(-b * (x_data - c))) + d
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
                "success": True,
                "status": 0,
                "message": "",
                "num_function_calls": getattr(res, "nfev", 0) if model_type != "polynomial" else 0,
                "final_cost": float(np.sum(residuals**2)),
            },
        }

    def _create_residual_function(
        self,
        problem: Dict[str, Any],
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
        """
        Return a residual function and an initial guess array
        for non‑linear models.
        """
        model_type = problem["model_type"]

        if model_type == "exponential":

            def r(p):
                a, b, c = p
                return y_data - (a * _safe_exp(b * x_data) + c)

            guess = np.array([1.0, 0.05, 0.0], dtype=np.float64)

        elif model_type == "logarithmic":

            def r(p):
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)

            guess = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)

        elif model_type == "sigmoid":

            def r(p):
                a, b, c, d = p
                return y_data - (a / (1 + _safe_exp(-b * (x_data - c))) + d)

            guess = np.array(
                [3.0, 0.5, np.median(x_data), 0.0], dtype=np.float64
            )

        elif model_type == "sinusoidal":

            def r(p):
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)

            guess = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float64)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return r, guess