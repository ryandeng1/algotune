from typing import Any, Dict
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Fast weighted linear regression approximating ODR."""
        # Convert data to NumPy arrays
        x = np.asarray(problem["x"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        sy = np.asarray(problem["sy"], dtype=np.float64)

        # Construct weighted least‑squares system:
        #   minimize sum((y - (b0*x + b1))**2 / sy**2)
        #   equivalent to least‑squares on weighted matrix
        w = 1.0 / sy
        X = np.vstack((x * w, w)).T
        y_w = y * w

        # Solve: X @ beta = y_w
        beta, *_ = np.linalg.lstsq(X, y_w, rcond=None)

        return {"beta": beta.tolist()}