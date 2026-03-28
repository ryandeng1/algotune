from typing import Any, Dict, List
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, np.ndarray | float]) -> Dict[str, List[float]]:
        sol = self._solve(problem, debug=False)
        if sol.success:
            return {k: sol.y[:, -1].tolist() for k in problem}
        else:
            raise RuntimeError(f"Solver failed: {sol.message}")