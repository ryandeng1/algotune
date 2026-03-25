import numpy as np
from scipy.optimize import least_squares

class Solver:
    def solve(self, problem):
        x_data = np.array(problem['x_data'])
        y_data = np.array(problem['y_data'])
        model_type = problem['model_type']
        degree = problem.get('degree', None)

        if model_type == 'polynomial':
            V = np.vander(x_data, degree + 1, increasing=True)
            params, _, _, _ = np.linalg.lstsq(V, y_data, rcond=None)
            return {'params': params.tolist()}
        else:
            def residual_exp(p):
                a, b, c = p
                exp_vals = np.exp(np.clip(b * x_data, -50.0, 50.0))
                return y_data - (a * exp_vals + c)

            def residual_log(p):
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)

            def residual_sigmoid(p):
                a, b, c, d = p
                exp_vals = np.exp(np.clip(-b * (x_data - c), -50.0, 50.0))
                return y_data - (a / (1 + exp_vals) + d)

            def residual_sin(p):
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)

            residual_funcs = {
                'exponential': residual_exp,
                'logarithmic': residual_log,
                'sigmoid': residual_sigmoid,
                'sinusoidal': residual_sin
            }

            if model_type in residual_funcs:
                residual_func = residual_funcs[model_type]
                guess = None
                if model_type == 'exponential':
                    guess = np.array([1.0, 0.05, 0.0])
                elif model_type == 'logarithmic':
                    guess = np.array([1.0, 1.0, 1.0, 0.0])
                elif model_type == 'sigmoid':
                    guess = np.array([3.0, 0.5, np.median(x_data), 0.0])
                else:  # 'sinusoidal'
                    guess = np.array([2.0, 1.0, 0.0, 0.0])

                res = least_squares(residual_func, guess, method='trf', max_nfev=10000)
                params_opt = res.x
                residuals = res.fun
                mse = np.mean(residuals**2)
                return {
                    'params': params_opt.tolist(),
                    'residuals': residuals.tolist(),
                    'mse': float(mse),
                    'convergence_info': {
                        'success': res.success,
                        'status': res.status,
                        'message': res.message,
                        'num_function_calls': int(res.nfev),
                        'final_cost': float(np.sum(residuals**2))
                    }
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")
