from typing import Dict, List, Any
import numpy as np
import scipy.linalg

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List[List[complex]]]]:
        A = problem["matrix"]
        try:
            X, _ = scipy.linalg.sqrtm(A, disp=False)
        except Exception:
            return {"sqrtm": {"X": []}}
        return {"sqrtm": {"X": X.tolist()}}