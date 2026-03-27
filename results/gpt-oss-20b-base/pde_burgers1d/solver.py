import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        # Preserve original behavior: call the underlying solver implementation.
        sol = self._solve(problem, debug=False)

        if sol.success:
            # Convert final state to list
            return sol.y[:, -1].tolist()
        raise RuntimeError(f"Solver failed: {sol.message}")