import numpy as np
from scipy.optimize import least_squares

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        x_data = np.asarray(problem["x_data"])
        y_data = np.asarray(problem["y_data"])
        model_type = problem["model_type"]
        degree = problem.get("degree", None)
        
        def safe_exp(z):
            return np.exp(np.clip(z, -50.0, 50.0))
        
        if model_type == "polynomial":
            params = np.polyfit(x_data, y_data, degree)
            y_fit = np.polyval(params, x_data)
            residuals = y_data - y_fit
            mse = np.mean(residuals**2)
            return {
                "params": params.tolist(),
                "residuals": residuals.tolist(),
                "mse": mse,
                "convergence_info": {
                    "success": True,
                    "status": 0,
                    "message": "Exact solution found",
                    "num_function_calls": 1,
                    "final_cost": float(np.sum(residuals**2))
                }
            }
        
        else:
            def residual(p):
                if model_type == "exponential":
                    a, b, c = p
                    return y_data - (a * safe_exp(b * x_data) + c)
                elif model_type == "logarithmic":
                    a, b, c, d = p
                    return y_data - (a * np.log(b * x_data + c) + d)
                elif model_type == "sigmoid":
                    a, b, c, d = p
                    return y_data - (a / (1 + safe_exp(-b * (x_data - c))) + d)
                elif model_type == "sinusoidal":
                    a, b, c, d = p
                    return y_data - (a * np.sin(b * x_data + c) + d)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            
            if model_type == "exponential":
                guess = np.array([1.0, 0.05, 0.0])
            elif model_type == "logarithmic":
                guess = np.array([1.0, 1.0, 1.0, 0.0])
            elif model_type == "sigmoid":
                guess = np.array([3.0, 0.5, np.median(x_data), 0.0])
            elif model_type == "sinusoidal":
                guess = np.array([2.0, 1.0, 0.0, 0.0])
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            result = least_squares(
                residual,
                guess,
                method='trf',
                max_nfev=10000
            )
            
            params_opt = result.x
            
            if model_type == "exponential":
                a, b, c = params_opt
                y_fit = a * safe_exp(b * x_data) + c
            elif model_type == "logarithmic":
                a, b, c, d = params_opt
                y_fit = a * np.log(b * x_data + c) + d
            elif model_type == "sigmoid":
                a, b, c, d = params_opt
                y_fit = a / (1 + safe_exp(-b * (x_data - c))) + d
            elif model_type == "sinusoidal":
                a, b, c, d = params_opt
                y_fit = a * np.sin(b * x_data + c) + d
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            residuals = y_data - y_fit
            mse = np.mean(residuals**2)
            
            convergence_info = {
                "success": result.success,
                "status": result.status,
                "message": result.message,
                "num_function_calls": int(result.nfev),
                "final_cost": float(np.sum(residuals**2))
            }
            
            return {
                "params": params_opt.tolist(),
                "residuals": residuals.tolist(),
                "mse": mse,
                "convergence_info": convergence_info
            }
