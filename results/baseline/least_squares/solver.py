from collections.abc import Callable
from typing import Any
import numpy as np
from scipy.optimize import leastsq


def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        residual, guess = self._create_residual_function(problem)
        params_opt, cov_x, info, mesg, ier = leastsq(residual, guess, full_output=True, maxfev=10000)
        x_data = np.asarray(problem['x_data'])
        y_data = np.asarray(problem['y_data'])
        model_type = problem['model_type']
        if model_type == 'polynomial':
            y_fit = np.polyval(params_opt, x_data)
        elif model_type == 'exponential':
            a, b, c = params_opt
            y_fit = a * _safe_exp(b * x_data) + c
        elif model_type == 'logarithmic':
            a, b, c, d = params_opt
            y_fit = a * np.log(b * x_data + c) + d
        elif model_type == 'sigmoid':
            a, b, c, d = params_opt
            y_fit = a / (1 + _safe_exp(-b * (x_data - c))) + d
        else:
            a, b, c, d = params_opt
            y_fit = a * np.sin(b * x_data + c) + d
        residuals = y_data - y_fit
        mse = float(np.mean(residuals ** 2))
        return {'params': params_opt.tolist(), 'residuals': residuals.tolist(), 'mse': mse, 'convergence_info': {'success': ier in {1, 2, 3, 4}, 'status': int(ier), 'message': mesg, 'num_function_calls': int(info['nfev']), 'final_cost': float(np.sum(residuals ** 2))}}

    def _create_residual_function(self, problem: dict[str, Any]) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
        x_data = np.asarray(problem['x_data'])
        y_data = np.asarray(problem['y_data'])
        model_type = problem['model_type']
        if model_type == 'polynomial':
            deg = problem['degree']

            def r(p):
                return y_data - np.polyval(p, x_data)
            guess = np.ones(deg + 1)
        elif model_type == 'exponential':

            def r(p):
                a, b, c = p
                return y_data - (a * _safe_exp(b * x_data) + c)
            guess = np.array([1.0, 0.05, 0.0])
        elif model_type == 'logarithmic':

            def r(p):
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)
            guess = np.array([1.0, 1.0, 1.0, 0.0])
        elif model_type == 'sigmoid':

            def r(p):
                a, b, c, d = p
                return y_data - (a / (1 + _safe_exp(-b * (x_data - c))) + d)
            guess = np.array([3.0, 0.5, np.median(x_data), 0.0])
        elif model_type == 'sinusoidal':

            def r(p):
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)
            guess = np.array([2.0, 1.0, 0.0, 0.0])
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        return (r, guess)
