import numpy as np

class Solver:
    def solve(self, problem: dict[str, any]) -> list[float]:
        X, y = problem.get("X", np.array([])), problem.get("y", np.array([]))
        if X.size == 0 or y.size == 0:
            return []
        try:
            # Solve the least‑squares problem X * beta = y
            coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            # Fallback: return a zero vector with correct dimensionality
            dims = X.shape[1] if X.ndim > 1 else 0
            return [0.0] * dims
        else:
            return coeff.tolist()