from typing import Any
import numpy as np
from scipy.optimize import leastsq


def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    return np.exp(np.clip(z, -50.0, 50.0))


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        residual, guess = self._create_residual_function(problem)
        params_opt, cov_x, info, mesg, ier = leastsq(
            residual, guess, full_output=True, maxfev=10000
        )

        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]

        model_functions = {
            "polynomial": lambda params, x: np.polyval(params, x),
            "exponential": lambda params, x: params[0] * _safe_exp(params[1] * x) + params[2],
            "logarithmic": lambda params, x: params[0] * np.log(params[1] * x + params[2]) + params[3],
            "sigmoid": lambda params, x: params[0] / (1 + _safe_exp(-params[1] * (x - params[2]))) + params[3],
            "sinusoidal": lambda params, x: params[0] * np.sin(params[1] * x + params[2]) + params[3],
        }

        y_fit = model_functions[model_type](params_opt, x_data)

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