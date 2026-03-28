import numpy as np
from scipy.optimize import leastsq

# Avoid importing the whole module for speed
def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Local copies to avoid attribute lookups
        x = np.asarray(problem['x_data'])
        y = np.asarray(problem['y_data'])
        model_type = problem['model_type']

        # Construct residual function and initial guess
        if model_type == 'polynomial':
            deg = problem['degree']

            def residual(p):
                return y - np.polyval(p, x)

            guess = np.ones(deg + 1)

        elif model_type == 'exponential':
            def residual(p):
                a, b, c = p
                return y - (a * _safe_exp(b * x) + c)

            guess = np.array([1.0, 0.05, 0.0])

        elif model_type == 'logarithmic':
            def residual(p):
                a, b, c, d = p
                return y - (a * np.log(b * x + c) + d)

            guess = np.array([1.0, 1.0, 1.0, 0.0])

        elif model_type == 'sigmoid':
            def residual(p):
                a, b, c, d = p
                return y - (a / (1 + _safe_exp(-b * (x - c))) + d)

            guess = np.array([3.0, 0.5, np.median(x), 0.0])

        elif model_type == 'sinusoidal':
            def residual(p):
                a, b, c, d = p
                return y - (a * np.sin(b * x + c) + d)

            guess = np.array([2.0, 1.0, 0.0, 0.0])

        else:
            raise ValueError(f'Unknown model type: {model_type}')

        # Perform least‑squares optimization
        p_opt, cov_x, info, mesg, ier = leastsq(
            residual, guess, full_output=True, maxfev=10000
        )

        # Compute fitted values
        if model_type == 'polynomial':
            y_fit = np.polyval(p_opt, x)
        elif model_type == 'exponential':
            a, b, c = p_opt
            y_fit = a * _safe_exp(b * x) + c
        elif model_type == 'logarithmic':
            a, b, c, d = p_opt
            y_fit = a * np.log(b * x + c) + d
        elif model_type == 'sigmoid':
            a, b, c, d = p_opt
            y_fit = a / (1 + _safe_exp(-b * (x - c))) + d
        else:  # sinusoidal
            a, b, c, d = p_opt
            y_fit = a * np.sin(b * x + c) + d

        residuals = y - y_fit
        mse = float(np.mean(residuals ** 2))
        cost = float(np.sum(residuals ** 2))

        return {
            'params': p_opt.tolist(),
            'residuals': residuals.tolist(),
            'mse': mse,
            'convergence_info': {
                'success': ier in {1, 2, 3, 4},
                'status': int(ier),
                'message': mesg,
                'num_function_calls': int(info['nfev']),
                'final_cost': cost,
            },
        }