from typing import Any
import numpy as np
from scipy.optimize import leastsq

# --- helper ---------------------------------------------------------------

def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

# --- solver ---------------------------------------------------------------

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Create residual function (implementation omitted)
        residual, guess = self._create_residual_function(problem)

        # Run least‑squares
        params_opt, cov_x, info, mesg, ier = leastsq(
            residual, guess, full_output=True, maxfev=10000
        )

        # Prepare data
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model = problem["model_type"]

        # Compute fitted values
        if model == "polynomial":
            y_fit = np.polyval(params_opt, x_data)
        elif model == "exponential":
            a, b, c = params_opt
            y_fit = a * _safe_exp(b * x_data) + c
        elif model == "logarithmic":
            a, b, c, d = params_opt
            y_fit = a * np.log(b * x_data + c) + d
        elif model == "sigmoid":
            a, b, c, d = params_opt
            y_fit = a / (1 + _safe_exp(-b * (x_data - c))) + d
        else:  # sinusoidal
            a, b, c, d = params_opt
            y_fit = a * np.sin(b * x_data + c) + d

        residuals = y_data - y_fit
        mse = float(np.mean(residuals ** 2))

        return {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": ier in {1, 2, 3, 4},
                "status": int(ier),
                "message": mesg,
                "num_function_calls": int(info["nfev"]),
                "final_cost": float(np.sum(residuals ** 2)),
            },
        }

    # Placeholder for the actual residual function creation
    def _create_residual_function(self, problem: dict[str, Any]):
        # This method must be implemented elsewhere; included here
        # to keep the signature consistent.
        raise NotImplementedError