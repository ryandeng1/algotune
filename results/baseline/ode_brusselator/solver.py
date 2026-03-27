from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray | float]) -> dict[str, list[float]]:
        sol = self._solve(problem, debug=False)

        # Extract final state
        if sol.success:
            return sol.y[:, -1]  # Get final state
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")
