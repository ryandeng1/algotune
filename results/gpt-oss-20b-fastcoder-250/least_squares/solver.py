from typing import Any

import numpy as np
from scipy.optimize import leastsq


def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        residual, guess = self._create_residual_function(problem)

        params_opt, cov_x, info, mesg, ier = leastsq(
            residual, guess, full_output=True, maxfev=10000
        )

        x = np.asanyarray(problem["x_data"])
        y = np.asanyarray(problem["y_data"])
        mtype = problem["model_type"]

        if mtype == "polynomial":
            y_fit = np.polyval(params_opt, x)
        elif mtype == "exponential":
            a, b, c = params_opt
            y_fit = a * _safe_exp(b * x) + c
        elif mtype == "logarithmic":
            a, b, c, d = params_opt
            y_fit = a * np.log(b * x + c) + d
        elif mtype == "sigmoid":
            a, b, c, d = params_opt
            y_fit = a / (1 + _safe_exp(-b * (x - c))) + d
        else:  # sinusoidal
            a, b, c, d = params_opt
            y_fit = a * np.sin(b * x + c) + d

        residuals = y - y_fit
        mse = float(np.mean(residuals * residuals))

        return {
            "params": params_opt.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": ier in {1, 2, 3, 4},
                "status": int(ier),
                "message": mesg,
                "num_function_calls": int(info["nfev"]),
                "final_cost": float(np.sum(residuals * residuals)),
            },
        }

    # --------------------------------------------------------------------
    # The helper method below is unchanged except for minor inlining
    # --------------------------------------------------------------------
    def _create_residual_function(self, problem: dict[str, Any]):
        x = np.asanyarray(problem["x_data"])
        y = np.asanyarray(problem["y_data"])
        mtype = problem["model_type"]

        if mtype == "polynomial":
            def residual(params: np.ndarray):
                return y - np.polyval(params, x)
            guess = np.ones(x.shape[0] + 1)
        elif mtype == "exponential":
            def residual(params: np.ndarray):
                a, b, c = params
                return y - (a * _safe_exp(b * x) + c)
            guess = [1.0, 0.0, y.mean()]
        elif mtype == "logarithmic":
            def residual(params: np.ndarray):
                a, b, c, d = params
                return y - (a * np.log(b * x + c) + d)
            guess = [1.0, 1.0, 1.0, y.mean()]
        elif mtype == "sigmoid":
            def residual(params: np.ndarray):
                a, b, c, d = params
                return y - (a / (1 + _safe_exp(-b * (x - c))) + d)
            guess = [1.0, 1.0, x.mean(), 0.0]
        else:  # sinusoidal
            def residual(params: np.ndarray):
                a, b, c, d = params
                return y - (a * np.sin(b * x + c) + d)
            guess = [1.0, 1.0, 0.0, 0.0]

        return residual, np.asarray(guess, dtype=float)