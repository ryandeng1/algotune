import numpy as np
from scipy.optimize import leastsq
from collections.abc import Callable
from typing import Any, Dict, Tuple

def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:
    def _create_residual_function(self, problem: Dict[str, Any]) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
        x = np.asarray(problem['x_data'])
        y = np.asarray(problem['y_data'])
        mt = problem['model_type']

        # Local variables for faster access
        if mt == 'polynomial':
            deg = problem['degree']
            def r(p: np.ndarray) -> np.ndarray:
                return y - np.polyval(p, x)
            guess = np.ones(deg + 1)
        elif mt == 'exponential':
            def r(p: np.ndarray) -> np.ndarray:
                a, b, c = p
                return y - (a * _safe_exp(b * x) + c)
            guess = np.array([1.0, 0.05, 0.0])
        elif mt == 'logarithmic':
            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a * np.log(b * x + c) + d)
            guess = np.array([1.0, 1.0, 1.0, 0.0])
        elif mt == 'sigmoid':
            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a / (1 + _safe_exp(-b * (x - c))) + d)
            guess = np.array([3.0, 0.5, np.median(x), 0.0])
        elif mt == 'sinusoidal':
            def r(p: np.ndarray) -> np.ndarray:
                a, b, c, d = p
                return y - (a * np.sin(b * x + c) + d)
            guess = np.array([2.0, 1.0, 0.0, 0.0])
        else:
            raise ValueError(f'Unknown model type: {mt}')

        return r, guess

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        residual, guess = self._create_residual_function(problem)

        params_opt, cov_x, info, mesg, ier = leastsq(
            residual, guess, full_output=True, maxfev=10000)

        x = np.asarray(problem['x_data'])
        y = np.asarray(problem['y_data'])
        mt = problem['model_type']

        if mt == 'polynomial':
            y_fit = np.polyval(params_opt, x)
        elif mt == 'exponential':
            a, b, c = params_opt
            y_fit = a * _safe_exp(b * x) + c
        elif mt == 'logarithmic':
            a, b, c, d = params_opt
            y_fit = a * np.log(b * x + c) + d
        elif mt == 'sigmoid':
            a, b, c, d = params_opt
            y_fit = a / (1 + _safe_exp(-b * (x - c))) + d
        else:  # sinusoidal
            a, b, c, d = params_opt
            y_fit = a * np.sin(b * x + c) + d

        residuals = y - y_fit
        mse = float(np.mean(residuals ** 2))

        return {
            'params': params_opt.tolist(),
            'residuals': residuals.tolist(),
            'mse': mse,
            'convergence_info': {
                'success': ier in {1, 2, 3, 4},
                'status': int(ier),
                'message': mesg,
                'num_function_calls': int(info['nfev']),
                'final_cost': float(np.sum(residuals ** 2))
            }
        }