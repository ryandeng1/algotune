# solver.py
from typing import Any, Dict, List
import numpy as np


class Solver:
    """
    Solves the L0‑pruning problem in linear time using NumPy’s fast argpartition.
    """

    @staticmethod
    def _prune(v: np.ndarray, k: int) -> np.ndarray:
        """
        Keep only the k entries of v with the largest absolute values
        (preserving their original signs), all other entries are set to zero.
        """
        n = v.size
        if k <= 0:
            return np.zeros_like(v)
        if k >= n:
            return v.copy()

        # Quick selection of the k largest absolute values.
        # argpartition returns indices that would sort the array; we only need the first k.
        # Using a negative sign gives us the k positions with largest |v|.
        idx = np.argpartition(-np.abs(v), k - 1)[:k]
        ret = np.zeros_like(v)
        ret[idx] = v[idx]
        return ret

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the L0‑pruning task described in the paper.

        Parameters
        ----------
        problem:
            * 'v': array‑like with the weights.
            * 'k': integer, number of elements to keep.

        Returns
        -------
        dict
            {'solution': [float, ...]} where the list contains the pruned vector.
        """
        v = np.asarray(problem.get("v")).ravel()
        k = int(problem.get("k", 0))

        pruned = self._prune(v, k)

        return {"solution": pruned.tolist()}