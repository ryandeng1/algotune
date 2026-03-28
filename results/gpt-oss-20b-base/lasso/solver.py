from typing import Any, List
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[float]:
        """Fast least‑squares solution without external dependencies."""
        X = problem.get("X")
        y = problem.get("y")
        try:
            # Solve X @ beta = y via least squares
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return beta.tolist()
        except Exception:
            # In case of failure (e.g. singular matrix) return zero vector
            _, d = X.shape
            return np.zeros(d).tolist()