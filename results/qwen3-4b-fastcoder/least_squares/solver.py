from typing import Any
import numpy as np
from scipy.optimize import leastsq


def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))


class Solver:
    _MODEL_FUNCTIONS = {
        "polynomial": lambda p, x: np.polyval(p, x),
        "exponential": lambda p, x: p[0] * _safe_exp(p[1] * x) + p[2],
        "logarithmic": lambda p, x: p[0] * np.log(p[1] * x + p[2]) + p[3],
        "sigmoid": lambda p, x: p[0] / (1 + _safe_exp(-p[1] * (x - p[2]))) + p[3],
        "sinusoidal": lambda p, x: p[0] * np.sin(p[1] * x + p[2]) + p[3],
    }

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        residual, guess = self._create_residual_function(problem)
        params_opt, cov_x, info, mesg, ier = leastsq(
            residual, guess, full_output=True, maxfev=10000
        )

        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        model_func = Solver._MODEL_FUNCTIONS.get(model_type)
        if model_func is None:
            raise ValueError(f"Unknown model type: {model_type}")

        y_fit = model_func(params_opt, x_data)
        residuals = y_data - y_fit
        mse = float(np.mean(residuals**2))
        total_residuals_sq = np.sum(residuals**2)

        return {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": ier in {1, 2, 3, 4},
                "status": int(ier),
                "message": mesg,
                "num_function_calls": int(info["nfev"]),
                "final_cost": float(total_residuals_sq),
            },
        }