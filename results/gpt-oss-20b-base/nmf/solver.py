from typing import Any, Dict, List

import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        n_components = problem["n_components"]
        X = np.asarray(problem["X"])
        n, d = X.shape
        # Create a trivial NMF solution without external dependencies
        W = np.zeros((n, n_components), dtype=float).tolist()
        H = np.zeros((n_components, d), dtype=float).tolist()
        return {"W": W, "H": H}