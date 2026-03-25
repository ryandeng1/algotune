import numpy as np
from scipy.optimize import curve_fit, least_squares

# Helper functions for models
def _safe_exp(z):
    return np.exp(np.clip(z, -50, 50))

def poly_fit(x, y, deg):
    # Construct Vandermonde matrix and solve with linalg.lstsq
    A = np.vander(x, deg + 1, increasing=False)
    return np.linalg.lstsq(A, y, rcond=None)[0]

def exp_fun(x, a, b, c):
    return a * _safe_exp(b * x) + c

def log_fun(x, a, b, c, d):
    return a * np.log(b * x + c) + d

def sigmoid_fun(x, a, b, c, d):
    return a / (1 + _safe_exp(-b * (x - c))) + d

def sin_fun(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

class Solver:
    def solve(self, problem):
        x = np.asarray(problem["x_data"])
        y = np.asarray(problem["y_data"])
        mt = problem["model_type"]
        params = None

        if mt == "polynomial":
            deg = problem["degree"]
            params = poly_fit(x, y, deg)

        else:
            # Choose function and initial guess
            if mt == "exponential":
                func, p0 = exp_fun, [1.0, 0.05, 0.0]
            elif mt == "logarithmic":
                func, p0 = log_fun, [1.0, 1.0, 1.0, 0.0]
            elif mt == "sigmoid":
                func, p0 = sigmoid_fun, [3.0, 0.5, np.median(x), 0.0]
            elif mt == "sinusoidal":
                func, p0 = sin_fun, [2.0, 1.0, 0.0, 0.0]
            else:
                raise ValueError(f"Unknown model type {mt}")

            # Perform curve fitting with least_squares
            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
            params = popt

        # Return just params as list
        return {"params": params.tolist()}
