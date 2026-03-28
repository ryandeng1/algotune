import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fast weighted linear regression (ignoring x uncertainties)."""
        x = problem["x"]
        y = problem["y"]
        sy = problem["sy"]
        # Convert to numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        # Weighted least squares: minimize sum((y - (a*x+b))/sy)^2
        w = 1.0 / sy**2
        sqrt_w = np.sqrt(w)
        X = np.vstack([x, np.ones_like(x)]).T
        Xw = X * sqrt_w[:, None]
        yw = y * sqrt_w
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        return {"beta": beta.tolist()}