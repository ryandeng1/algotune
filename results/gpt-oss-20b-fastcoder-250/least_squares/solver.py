import numpy as np
from scipy.optimize import curve_fit
from typing import Any, Dict, List


def _safe_exp(z: np.ndarray | float) -> np.ndarray | float:
    """Exponentiation clipped to avoid overflow."""
    return np.exp(np.clip(z, -50.0, 50.0))


class Solver:
    def _model_func(self, model_type: str):
        if model_type == "polynomial":

            def f(np_in, *coef):
                # coef order : highest degree first
                return np.polyval(coef, np_in)

        elif model_type == "exponential":

            def f(x, a, b, c):
                return a * _safe_exp(b * x) + c

        elif model_type == "logarithmic":

            def f(x, a, b, c, d):
                return a * np.log(b * x + c) + d

        elif model_type == "sigmoid":

            def f(x, a, b, c, d):
                return a / (1 + _safe_exp(-b * (x - c))) + d

        elif model_type == "sinusoidal":

            def f(x, a, b, c, d):
                return a * np.sin(b * x + c) + d

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return f

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        x = np.asarray(problem["x_data"], dtype=float)
        y = np.asarray(problem["y_data"], dtype=float)
        mtype = problem["model_type"]

        if mtype == "polynomial":
            deg = problem["degree"]
            # numpy.polyfit is highly optimized
            coeffs = np.polyfit(x, y, deg)
            # no extra info needed
            return {"params": coeffs.tolist()}

        func = self._model_func(mtype)

        # Initial guess handling
        if mtype == "polynomial":
            p0 = np.ones(deg + 1)
        else:
            p0 = None

        # Use curve_fit which is fast and robust
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=5000)
        return {"params": popt.tolist()}
