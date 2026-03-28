import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        v = np.asarray(problem['v']).ravel()
        k = problem['k']

        # Subproblem solver using vectorisation
        def subproblem_sol(vn: np.ndarray, z: float) -> np.ndarray:
            mu = np.sort(vn, kind='mergesort')[::-1]  # descending
            cumsum = np.cumsum(mu)
            denom = np.arange(1, mu.size + 1)
            theta_candidates = (cumsum - z) / denom
            # Find first index where mu < theta
            mask = mu < theta_candidates
            idx = np.argmax(mask)
            if mask.any() and idx < mask.size:
                theta = theta_candidates[idx]
            else:
                theta = 0.0
            return np.maximum(vn - theta, 0.0)

        u = np.abs(v)
        b = subproblem_sol(u, k)
        new_v = b * np.sign(v)

        # Only keep non‑zero entries (pruning)
        pruned = np.zeros_like(v, dtype=float)
        pruned[new_v != 0] = new_v[new_v != 0]

        return {'solution': pruned.tolist()}