import numpy as np
from scipy.optimize import least_squares

def _safe_exp(z):
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:
    def solve(self, problem):
        model_type = problem["model_type"]
        x = np.asarray(problem["x_data"])
        y = np.asarray(problem["y_data"])

        # ------------------------------------------------------------------
        # Polynomial: use fast NumPy polyfit
        # ------------------------------------------------------------------
        if model_type == "polynomial":
            deg = problem["degree"]
            params = np.polyfit(x, y, deg)
            y_fit = np.polyval(params, x)
            residuals = y - y_fit
        else:
            # ------------------------------------------------------------------
            # Non‑polynomial models fit with least_squares
            # ------------------------------------------------------------------
            if model_type == "exponential":
                def residual(p):
                    a, b, c = p
                    return y - (a * _safe_exp(b * x) + c)
                guess = np.array([1.0, 0.05, 0.0])
            elif model_type == "logarithmic":
                def residual(p):
                    a, b, c, d = p
                    return y - (a * np.log(b * x + c) + d)
                guess = np.array([1.0, 1.0, 1.0, 0.0])
            elif model_type == "sigmoid":
                def residual(p):
                    a, b, c, d = p
                    return y - (a / (1 + _safe_exp(-b * (x - c))) + d)
                guess = np.array([3.0, 0.5, np.median(x), 0.0])
            elif model_type == "sinusoidal":
                def residual(p):
                    a, b, c, d = p
                    return y - (a * np.sin(b * x + c) + d)
                guess = np.array([2.0, 1.0, 0.0, 0.0])
            else:
                raise ValueError(f"Unsupported model: {model_type}")

            result = least_squares(residual, guess, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            params = result.x
            y_fit = residual(params) * -1  # residuals = y - fit, so fit = y - residual
            residuals = y - y_fit

        mse = float(np.mean(residuals**2))
        return {
            "params": params.tolist(),
            "residuals": residuals.tolist(),
            "mse": mse,
            "convergence_info": {
                "success": not np.isnan(params).any(),
                "status": int(result.success if model_type != "polynomial" else 1),
                "message": result.message if model_type != "polynomial" else "polynomial fit",
                "num_function_calls": int(result.nfev) if model_type != "polynomial" else 0,
                "final_cost": float(np.sum(residuals**2)),
            },
        }